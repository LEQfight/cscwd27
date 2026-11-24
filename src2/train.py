"""训练并验证正文 ValueReadout，入口为 python -m src2.train。"""

import argparse
from functools import partial
from itertools import islice
import json
from math import ceil
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from .data import collate, load_split
from .model import LastTokenBaseline, ValueReadout


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, predictions = [], []
    for batch in loader:
        labels.extend(batch.pop("labels").tolist())
        output = model(**{key: value.to(device) for key, value in batch.items()})
        predictions.append(output["logits"].argmax(dim=-1))
    predictions = torch.cat(predictions).cpu().tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], average="macro", zero_division=0,
    )
    return {"acc": float(accuracy_score(labels, predictions)), "macro_precision": float(precision),
            "macro_recall": float(recall), "macro_f1": float(f1)}


def train(model, train_loader, train_eval_loader, val_loader, device, args, tracker):
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=1e-5,
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, threshold=0.001,
        threshold_mode="abs", min_lr=min(args.lr, 1e-6),
    )
    accumulation = args.batch_size // args.micro_batch
    warmup_epochs = min(2, args.epochs)
    warmup_steps = warmup_epochs * ceil(len(train_loader) / accumulation)
    checkpoint_path = args.run_dir / "checkpoints" / "best.pt"
    best_f1 = patience_reference = -float("inf")
    stale_epochs = step = 0

    with (args.run_dir / "metrics" / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            batches = iter(train_loader)
            for _ in range(0, len(train_loader), accumulation):
                # 梯度按样本数平均，末尾不足一个有效 batch 也能正确更新。
                group = list(islice(batches, accumulation))
                samples = sum(batch["labels"].numel() for batch in group)
                optimizer.zero_grad(set_to_none=True)
                update_loss = 0.0
                for batch in group:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    loss = model(**batch)["loss"] * batch["labels"].numel() / samples
                    loss.backward()
                    update_loss += loss.detach()

                step += 1
                if step <= warmup_steps:
                    optimizer.param_groups[0]["lr"] = args.lr * step / warmup_steps
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                values = {"train/loss": update_loss.item(), "train/learning_rate": lr}
                if step % args.eval_every == 0:
                    for split, loader in (("train", train_eval_loader), ("val", val_loader)):
                        values.update({f"{split}/{key}": value
                                       for key, value in evaluate(model, loader, device).items()})
                    model.train()
                history.write(json.dumps({"epoch": epoch, "update_step": step, **values}) + "\n")
                history.flush()
                tracker.log(values, step=step)

            # 轮末验证用于保存最佳 checkpoint、调学习率和早停。
            scores = evaluate(model, val_loader, device)
            f1 = scores["macro_f1"]
            history.write(json.dumps({"epoch_end": epoch, "update_step": step, "validation": scores}) + "\n")
            history.flush()
            tracker.log({f"epoch_val/{key}": value for key, value in scores.items()}, step=step)
            print(f"epoch={epoch} val={scores}", flush=True)
            if f1 > best_f1:
                best_f1 = f1
                checkpoint_tmp = checkpoint_path.with_suffix(".tmp")
                torch.save({"adapter": model.adapter_state(), "epoch": epoch,
                            "model": str(args.model.resolve()), "experiment": args.experiment,
                            "validation": scores}, checkpoint_tmp)
                checkpoint_tmp.replace(checkpoint_path)
            if epoch >= warmup_epochs:
                plateau.step(f1)
                if f1 - patience_reference >= 0.001:
                    patience_reference = f1
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= args.patience:
                    break

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_adapter(best["adapter"])
    return {"best_epoch": best["epoch"], "epochs_completed": epoch,
            "validation": best["validation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", required=True, help="Existing SwanLab workspace")
    parser.add_argument("--experiment", choices=("A", "B1", "B2", "B3", "B4", "B5", "value-readout"),
                        default="value-readout")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Trade training speed for activation recomputation")
    parser.add_argument("--max-text-tokens", type=int, default=1533)
    parser.add_argument("--eval-every", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.micro_batch, args.eval_batch, args.max_text_tokens,
           args.eval_every, args.epochs, args.patience) < 1 or args.lr <= 0:
        parser.error("Training parameters must be positive")
    if args.batch_size % args.micro_batch:
        parser.error("--micro-batch must divide --batch-size")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a CUDA device with BF16 support")

    import swanlab

    torch.manual_seed(args.seed)
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    backbone = AutoModel.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16)
    if args.experiment == "A":
        model = LastTokenBaseline(backbone)
    else:
        feature = {"value-readout": "value", "B1": "value", "B2": "hidden", "B3": "hidden-mean",
                   "B4": "value-mean", "B5": "hidden-mean"}[args.experiment]
        model = ValueReadout(backbone, feature=feature, last_only=args.experiment in ("B4", "B5"),
                             gradient_checkpointing=args.gradient_checkpointing)
    if args.max_text_tokens + model.readout_tokens > backbone.config.max_position_embeddings:
        parser.error("The text and readouts exceed the context window")
    model.to(device)

    train_rows = load_split(args.data_dir / "train.pkl", tokenizer, args.max_text_tokens)
    val_rows = load_split(args.data_dir / "val.pkl", tokenizer, args.max_text_tokens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collate_fn = partial(collate, pad_id=pad_id)
    train_loader = DataLoader(train_rows, batch_size=args.micro_batch, shuffle=True,
                              collate_fn=collate_fn, generator=torch.Generator().manual_seed(args.seed))
    # 按长度排序评估批次，减少 padding。训练批次正常打乱。
    train_eval_loader, val_loader = [
        DataLoader(sorted(rows, key=lambda row: len(row[0])),
                   batch_size=args.eval_batch, collate_fn=collate_fn)
        for rows in (train_rows, val_rows)
    ]

    project = "readout_token"
    swanlab.Api().project(f"{args.workspace}/{project}")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("checkpoints", "metrics", "swanlab"):
        (args.run_dir / folder).mkdir()
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(train_samples=len(train_rows), val_samples=len(val_rows),
                  readout_tokens=model.readout_tokens,
                  weight_decay=1e-5, warmup_epochs=min(2, args.epochs),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.experiment in ("value-readout", "B1", "B4"):
        config["value_size"] = model.value_size
    elif args.experiment in ("B2", "B3", "B5"):
        config["hidden_size"] = backbone.config.hidden_size
    if args.experiment != "A":
        config.update(readout_layers=list(model.readout_layers), readout_positions=model.readout_positions)
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


if __name__ == "__main__":
    main()
