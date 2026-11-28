"""训练并验证Weibo21 readout 分类器，支持单卡和 torchrun 多卡。"""

import argparse
from contextlib import nullcontext
from functools import partial
from itertools import islice
import json
from math import ceil
import os
from pathlib import Path

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer

from .data import collate, evaluation_loader, load_split
from .model import READOUT_TOKENS, HiddenReadout, LastTokenBaseline
from .training_state import capture_rng, link_best, protocol, restore_training, save_checkpoint


@torch.inference_mode()
def evaluate(model, loader, device, *, auroc=False):
    model.eval()
    # 每张卡只统计自己的样本，再合并二分类混淆矩阵。
    counts = torch.zeros(4, dtype=torch.long, device=device)
    ranked = []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        output = model(**{key: value.to(device) for key, value in batch.items()})
        logits = output["logits"]
        predictions = logits.argmax(dim=-1)
        counts.scatter_add_(0, 2 * labels + predictions, torch.ones_like(labels))
        if auroc:
            # Fake 类的 logit 差与二分类概率具有相同排序。
            ranked.append(torch.stack((labels, logits[:, 1].float() - logits[:, 0].float()), dim=1))
    if dist.is_initialized():
        dist.all_reduce(counts)
    weights = counts.cpu().tolist()
    labels, predictions = [0, 0, 1, 1], [0, 1, 0, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], sample_weight=weights, average="macro", zero_division=0,
    )
    metrics = {"acc": float(accuracy_score(labels, predictions, sample_weight=weights)),
               "macro_precision": float(precision),
               "macro_recall": float(recall), "macro_f1": float(f1)}
    if auroc:
        local = torch.cat(ranked).cpu().tolist() if ranked else []
        shards = [local]
        if dist.is_initialized():
            shards = [None] * dist.get_world_size()
            # 集合通信的内部缓冲区须允许在推理模式之外写入。
            with torch.inference_mode(False):
                dist.all_gather_object(shards, local)
        target, score = zip(*(row for shard in shards for row in shard))
        metrics["auroc"] = float(roc_auc_score(target, score))
    return metrics


def train(model, train_loader, train_eval_loader, val_loader, device, args, tracker):
    distributed = isinstance(model, DistributedDataParallel)
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    adapter_model = model.module if distributed else model
    save_state = getattr(args, "save_training_state", False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=1e-5,
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, threshold=0.001,
        threshold_mode="abs", min_lr=min(args.lr, 1e-6),
    )
    accumulation = args.batch_size // (world_size * args.micro_batch)
    warmup_epochs = min(2, args.epochs)
    warmup_steps = warmup_epochs * ceil(len(train_loader) / accumulation)
    checkpoint_path = args.run_dir / "checkpoints" / "best.pt"
    latest_path = checkpoint_path.with_name("last.pt")
    best_f1 = patience_reference = -float("inf")
    best = None
    stale_epochs = step = previous_epoch = 0
    if getattr(args, "resume_from", None):
        previous_epoch, step, best, patience_reference, stale_epochs = restore_training(
            args, adapter_model, optimizer, plateau, train_loader, device, rank, world_size, checkpoint_path,
        )
        best_f1 = best["validation"]["macro_f1"]

    history_path = args.run_dir / "metrics" / "history.jsonl"
    history_file = history_path.open("w", encoding="utf-8") if rank == 0 else nullcontext()
    with history_file as history:
        for epoch in range(previous_epoch + 1, args.epochs + 1):
            model.train()
            if distributed:
                train_loader.sampler.set_epoch(epoch)
            batches = iter(train_loader)
            total_samples = len(train_loader.dataset)
            local_remaining = len(range(rank, total_samples, world_size))
            for batch_index in range(0, len(train_loader), accumulation):
                group = list(islice(batches, accumulation))
                # 分片补齐的样本以零权重参与同步，末尾按实际的全局样本数平均。
                samples = min(args.batch_size, total_samples - batch_index * args.micro_batch * world_size)
                optimizer.zero_grad(set_to_none=True)
                update_loss = torch.zeros((), device=device)
                for index, batch in enumerate(group):
                    count = min(local_remaining, batch["labels"].numel())
                    local_remaining -= count
                    batch = {key: value[:max(1, count)].to(device) for key, value in batch.items()}
                    sync = model.no_sync() if distributed and index < len(group) - 1 else nullcontext()
                    with sync:
                        loss = model(**batch)["loss"] * (count / samples)
                        # DDP 对各卡梯度取平均，乘回卡数得到全局样本均值。
                        (loss * world_size).backward()
                    update_loss += loss.detach()
                if distributed:
                    dist.reduce(update_loss, dst=0)

                step += 1
                if step <= warmup_steps:
                    optimizer.param_groups[0]["lr"] = args.lr * step / warmup_steps
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                values = {"train/loss": update_loss.item(), "train/learning_rate": lr} if rank == 0 else {}
                if step % args.eval_every == 0:
                    for split, loader in (("train", train_eval_loader), ("val", val_loader)):
                        values.update({f"{split}/{key}": value
                                       for key, value in evaluate(adapter_model, loader, device).items()})
                    model.train()
                if history is not None:
                    history.write(json.dumps({"epoch": epoch, "update_step": step, **values}) + "\n")
                    history.flush()
                    tracker.log(values, step=step)

            # 轮末验证用于保存最佳 checkpoint、调学习率和早停。
            scores = evaluate(adapter_model, val_loader, device)
            f1 = scores["macro_f1"]
            if history is not None:
                history.write(json.dumps({"epoch_end": epoch, "update_step": step, "validation": scores}) + "\n")
                history.flush()
                tracker.log({f"epoch_val/{key}": value for key, value in scores.items()}, step=step)
                print(f"epoch={epoch} val={scores}", flush=True)
            current = {"epoch": epoch, "model": str(args.model.resolve()), "experiment": args.experiment,
                       "head": args.head, "validation": scores}
            improved = f1 > best_f1
            if improved:
                best_f1, best = f1, current
            if epoch >= warmup_epochs:
                plateau.step(f1)
                if f1 - patience_reference >= 0.001:
                    patience_reference = f1
                    stale_epochs = 0
                else:
                    stale_epochs += 1

            # 在调度器和早停状态更新后保存，各卡随机数集中到主进程。
            rng = capture_rng(train_loader, device, world_size) if save_state else None
            if rank == 0 and (save_state or improved):
                checkpoint = {**current, "adapter": adapter_model.adapter_state()}
                if save_state:
                    checkpoint["training_state"] = {
                        "protocol": protocol(args, world_size), "optimizer": optimizer.state_dict(),
                        "scheduler": plateau.state_dict(), "rng": rng, "update_step": step,
                        "best": best, "patience_reference": patience_reference, "stale_epochs": stale_epochs,
                    }
                    save_checkpoint(latest_path, checkpoint)
                    if improved:
                        link_best(latest_path, checkpoint_path)
                else:
                    save_checkpoint(checkpoint_path, checkpoint)
            if stale_epochs >= args.patience:
                break

    if rank != 0:
        return None
    return {"best_epoch": best["epoch"], "epochs_completed": epoch,
            "validation": best["validation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", required=True, help="Existing SwanLab workspace")
    parser.add_argument("--experiment", choices=("A", "B2"), default="B2")
    parser.set_defaults(dataset="weibo21", head="mlp")
    parser.add_argument("--readout-tokens", type=int, default=READOUT_TOKENS)
    parser.add_argument("--readout-pooling", choices=("mean", "geometric"), default="mean")
    parser.add_argument("--readout-pool-tail", type=int, choices=(1,), help="Use only the last readout")
    parser.add_argument("--match-3-token-init", action="store_true",
                        help="Match the original three-token readout and classifier initialization")
    parser.add_argument("--keep-readout-prefix-init", action="store_true",
                        help="Use the mean-pooling initialization order even for geometric pooling")
    parser.add_argument("--aligned-readout-init", action="store_true",
                        help="Initialize independent readouts in order with a shared prefix across lengths")
    parser.add_argument("--readout-curriculum", action="store_true",
                        help="Warm-start the readout prefix and initialize the two new slots to zero")
    parser.add_argument("--readout-parent", type=Path, help="Previous curriculum stage's best.pt")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16, help="Global samples per optimizer update")
    parser.add_argument("--micro-batch", type=int, default=4, help="Training samples per GPU per forward")
    parser.add_argument("--eval-batch", type=int, default=16, help="Evaluation samples per GPU per forward")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Trade training speed for activation recomputation")
    parser.add_argument("--max-text-tokens", type=int, default=1533)
    parser.add_argument("--eval-every", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--save-training-state", action="store_true", help="Save resumable best and last checkpoints")
    parser.add_argument("--resume-from", type=Path, help="Resume from an epoch checkpoint into a new run directory")
    parser.add_argument("--reset-early-stop", action="store_true", help="Clear the restored early-stop counter")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.readout_parent and not args.readout_curriculum:
        parser.error("--readout-parent requires --readout-curriculum")
    if args.readout_curriculum:
        if args.experiment != "B2" or args.readout_tokens < 2 or args.readout_tokens % 2:
            parser.error("Readout curriculum requires B2 with an even suffix and full or last-token pooling")
        if bool(args.readout_parent) != (args.readout_tokens > 2):
            parser.error("Stages above two tokens require the previous best.pt as --readout-parent")
    if args.reset_early_stop and not args.resume_from:
        parser.error("--reset-early-stop requires --resume-from")
    args.save_training_state = args.save_training_state or args.resume_from is not None
    if min(args.batch_size, args.micro_batch, args.eval_batch, args.max_text_tokens,
           args.eval_every, args.epochs, args.patience) < 1 or args.lr <= 0:
        parser.error("Training parameters must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.batch_size % (world_size * args.micro_batch):
        parser.error("GPU count times --micro-batch must divide --batch-size")
    if args.readout_tokens < 1:
        parser.error("Invalid readout token count for the selected initialization")
    if args.match_3_token_init and args.aligned_readout_init:
        parser.error("Choose one readout initialization method")
    if args.readout_pool_tail and (args.experiment != "B2" or args.readout_pooling != "mean"):
        parser.error("Last-token readout requires B2 without geometric weighting")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {local_rank} is unavailable")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a CUDA device with BF16 support")
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    try:
        run(args, parser, device, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(args, parser, device, world_size):
    rank = dist.get_rank() if world_size > 1 else 0
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    backbone = AutoModel.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16)
    if backbone.config.model_type != "qwen3":
        parser.error("This Weibo21 release uses a Qwen3 backbone")
    model = build_model(backbone, args.experiment, readout_tokens=args.readout_tokens,
                        gradient_checkpointing=args.gradient_checkpointing,
                        match_3_token_init=args.match_3_token_init, pooling=args.readout_pooling,
                        pool_tail=args.readout_pool_tail, aligned_readout_init=args.aligned_readout_init,
                        keep_readout_prefix_init=args.keep_readout_prefix_init)
    if args.readout_curriculum and not args.resume_from:
        parent_readout = None
        if args.readout_parent:
            parent = torch.load(args.readout_parent, map_location="cpu", weights_only=True, mmap=True)
            if parent.get("experiment") != "B2" or parent.get("model") != str(args.model.resolve()):
                parser.error("The curriculum checkpoint must be B2 with the same backbone")
            parent_config = json.loads((args.readout_parent.parents[1] / "config.json").read_text(encoding="utf-8"))
            if (parent_config["data_dir"] != str(args.data_dir.resolve())
                    or parent_config["dataset"] != args.dataset
                    or parent_config.get("readout_pooling", "mean") != args.readout_pooling
                    or parent_config.get("readout_pool_tail") != args.readout_pool_tail):
                parser.error("The curriculum checkpoint must use the same data and pooling")
            parent_readout = parent["adapter"]["readout"]
        model.initialize_readout_curriculum(parent_readout)
    if args.max_text_tokens + model.suffix_tokens > backbone.config.max_position_embeddings:
        parser.error("The text and readouts exceed the context window")
    model.to(device)

    train_rows, val_rows = [load_split(args.data_dir / f"{split}.pkl", tokenizer, args.max_text_tokens)
                            for split in ("train", "val")]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collate_fn = partial(collate, pad_id=pad_id)
    sampler = DistributedSampler(train_rows, seed=args.seed) if world_size > 1 else None
    train_loader = DataLoader(train_rows, batch_size=args.micro_batch, shuffle=sampler is None, sampler=sampler,
                              collate_fn=collate_fn, generator=torch.Generator().manual_seed(args.seed))
    train_eval_loader, val_loader = [
        evaluation_loader(rows, args.eval_batch, collate_fn, rank, world_size)
        for rows in (train_rows, val_rows)
    ]

    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(world_size=world_size, gradient_accumulation_steps=args.batch_size // (world_size * args.micro_batch),
                  train_samples=len(train_rows), val_samples=len(val_rows),
                  readout_tokens=model.readout_tokens, suffix_tokens=model.suffix_tokens,
                  weight_decay=1e-5, warmup_epochs=min(2, args.epochs),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.experiment == "B2":
        config.update(hidden_size=backbone.config.hidden_size, readout_layers=list(model.readout_layers),
                      readout_positions=model.readout_positions)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    if rank != 0:
        train(model, train_loader, train_eval_loader, val_loader, device, args, tracker=None)
        return

    import swanlab

    # 只有主进程创建目录并写入日志、SwanLab 和 checkpoint。
    project = "readout_token"
    swanlab.Api().project(f"{args.workspace}/{project}")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("checkpoints", "metrics", "swanlab"):
        (args.run_dir / folder).mkdir()
    run_name = f"{args.run_dir.parent.name}-{args.run_dir.name}"
    tracker = swanlab.init(project=project, workspace=args.workspace, name=run_name,
                           config=config, log_dir=str(args.run_dir / "swanlab"), mode="online",
                           on_online_error="raise")
    try:
        (args.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        result = train(model, train_loader, train_eval_loader, val_loader, device, args, tracker)
        result["swanlab_url"] = tracker.url
        (args.run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        swanlab.finish()


def build_model(backbone, experiment, *, readout_tokens, **options):
    if experiment == "A":
        if options.get("match_3_token_init"):
            # 对齐 A0 和 B2 的分类头随机数顺序。
            embeddings = backbone.get_input_embeddings().weight
            torch.randn(READOUT_TOKENS, embeddings.size(1), dtype=torch.float32, device=embeddings.device)
        return LastTokenBaseline(backbone)
    return HiddenReadout(backbone, readout_tokens=readout_tokens, **options)


if __name__ == "__main__":
    main()
