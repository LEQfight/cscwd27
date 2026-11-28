"""轮末保存与恢复优化器、学习率、选模和各卡 RNG 状态。"""

import os
from pathlib import Path
import random
import shutil

import torch
import torch.distributed as dist


RESUME_KEYS = (
    "model", "data_dir", "dataset", "experiment", "head", "baseline_readout",
    "full_finetune", "lora_rank", "lora_alpha", "lora_dropout", "lora_targets",
    "readout_tokens", "readout_pooling", "readout_repeats", "untie_readout_repeats",
    "readout_pool_tail", "match_3_token_init", "aligned_readout_init", "keep_readout_prefix_init",
    "lr", "batch_size", "micro_batch", "eval_batch", "seed", "max_text_tokens", "eval_every",
    "gradient_checkpointing",
)


def protocol(args, world_size):
    # 保留历史 Qwen3 checkpoint 中固定配置的默认值，以便直接续训。
    fixed = {"baseline_readout": "last", "full_finetune": False, "lora_rank": 0,
             "lora_alpha": 32, "lora_dropout": 0.05, "lora_targets": "qv",
             "readout_repeats": 1, "untie_readout_repeats": False}
    settings = {key: getattr(args, key, fixed.get(key)) for key in RESUME_KEYS}
    return {key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in settings.items()} | {"world_size": world_size, "warmup_epochs": min(2, args.epochs)}


def capture_rng(loader, device, world_size):
    local = {"torch": torch.get_rng_state(), "python": random.getstate(),
             "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
             "loader": loader.generator.get_state() if loader.generator is not None else None}
    if world_size == 1:
        return [local]
    states = [None] * world_size
    dist.all_gather_object(states, local)
    return states


def restore_training(args, model, optimizer, scheduler, loader, device, rank, world_size, checkpoint_path):
    source = args.resume_from.resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("training_state")
    if state is None:
        raise ValueError("This checkpoint does not contain optimizer state")
    if state["protocol"] != protocol(args, world_size):
        raise ValueError("Resume requires matching model, data, optimizer, pooling, batch and rank settings")
    if args.epochs <= checkpoint["epoch"]:
        raise ValueError("--epochs must be greater than the checkpoint epoch to extend training")

    best = state["best"]
    best_source = source if checkpoint["epoch"] == best["epoch"] else source.with_name("best.pt")
    if rank == 0:
        manifest = torch.load(best_source, map_location="cpu", weights_only=True, mmap=True)
        if manifest["epoch"] != best["epoch"] or manifest["validation"] != best["validation"]:
            raise ValueError("The saved best checkpoint does not match the resume state")
        shutil.copy2(best_source, checkpoint_path)

    model.load_adapter(checkpoint["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng = state["rng"][rank]
    torch.set_rng_state(rng["torch"])
    random.setstate(rng["python"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state(rng["cuda"], device)
    if rng["loader"] is not None:
        loader.generator.set_state(rng["loader"])
    return checkpoint["epoch"], state["update_step"], best, state["patience_reference"], (
        0 if args.reset_early_stop else state["stale_epochs"]
    )


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def link_best(latest, best):
    temporary = best.with_suffix(".link")
    temporary.unlink(missing_ok=True)
    os.link(latest, temporary)
    temporary.replace(best)
