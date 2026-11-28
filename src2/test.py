"""使用验证集选出的 best.pt 评估有标签的保留划分。"""

import argparse
import csv
from functools import partial
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer

from .data import collate, evaluation_loader, load_split
from .train import build_model, evaluate


def test_checkpoint(path, device, rank, world_size):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    checkpoint_path = path / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    experiment = config["experiment"]
    if (experiment not in ("A", "B2") or config.get("head") != "mlp"
            or config.get("dataset") != "weibo21" or config.get("full_finetune") or config.get("lora_rank")
            or config.get("baseline_readout", "last") != "last" or config.get("readout_repeats", 1) != 1
            or checkpoint["model"] != config["model"]):
        raise ValueError(f"Unsupported or incompatible Weibo21 checkpoint in {path}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    backbone = AutoModel.from_pretrained(config["model"], local_files_only=True, torch_dtype=torch.bfloat16)
    model = build_model(backbone, experiment, readout_tokens=config["readout_tokens"],
                        pooling=config.get("readout_pooling", "mean"), pool_tail=config.get("readout_pool_tail"))
    model.load_adapter(checkpoint["adapter"])
    model.to(device)
    test_path = Path(config["data_dir"]) / "test.pkl"
    rows = load_split(test_path, tokenizer, config["max_text_tokens"])
    if len(rows) != 1830:
        raise ValueError(f"Expected 1830 Weibo21 test samples, got {len(rows)}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    loader = evaluation_loader(rows, config["eval_batch"], partial(collate, pad_id=pad_id), rank, world_size)
    scores = evaluate(model, loader, device, auroc=True)
    if rank != 0:
        return None

    status = "completed"
    for name in ("result.json", "stopped-result.json"):
        result = path / name
        if result.is_file():
            status = json.loads(result.read_text(encoding="utf-8")).get("status", status)
            break
    with checkpoint_path.open("rb") as file:
        checkpoint_hash = hashlib.file_digest(file, "sha256").hexdigest()
    return {
        "model": Path(config["model"]).name, "head": "mlp", "readout_tokens": config["readout_tokens"],
        "baseline_readout": "last", "full_finetune": False, "lora_rank": 0,
        "dataset": "weibo21", "evaluation_split": "official_test",
        "suffix_tokens": model.suffix_tokens, "readout_repeats": 1,
        "readout_pooling": config.get("readout_pooling", "mean"),
        "epochs_cap": config["epochs"], "world_size": config["world_size"],
        "curriculum": config.get("readout_curriculum", False),
        "readout_pool_tail": config.get("readout_pool_tail"),
        "best_epoch": checkpoint["epoch"], "status": status, "test_samples": len(rows),
        "validation_acc": checkpoint["validation"]["acc"],
        "validation_macro_f1": checkpoint["validation"]["macro_f1"], **scores,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON object mapping method names to run directories")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    methods = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not methods:
        parser.error("The manifest is empty")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for method, path in methods.items():
            result = test_checkpoint(Path(path), device, rank, world_size)
            if rank == 0:
                row = dict(method=method, **result)
                records.append(row)
                (args.out / "test-results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                print(f"TEST RESULT {json.dumps(row)}", flush=True)
        if rank == 0:
            with (args.out / "test-results.csv").open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            print(f"TEST SWEEP COMPLETE {len(records)}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
