"""准备或运行 readout scaling。默认只打印命令。"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


COUNTS = tuple(range(0, 65, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--max-text-tokens", type=int, default=1533,
                        help="Shared text budget for all arms")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpus", type=int, default=2, help="Number of visible GPUs to use")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4, help="Validation Macro-F1 early-stop patience in epochs")
    parser.add_argument("--save-training-state", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--readout-counts", type=int, nargs="+", default=COUNTS)
    parser.add_argument("--match-3-token-init", action="store_true",
                        help="Match the historical Qwen3 three-token initialization")
    parser.add_argument("--readout-pooling", choices=("mean", "geometric", "last"), default="mean")
    parser.add_argument("--readout-curriculum", action="store_true")
    parser.add_argument("--curriculum-seed-run", type=Path, help="Warm-start from an existing even-token run")
    parser.add_argument("--baseline-run", type=Path, help="Include an existing A0 checkpoint in the final test")
    parser.add_argument("--test-run", type=Path, action="append", default=[],
                        help="Include a previously trained readout count in the final test")
    parser.add_argument("--execute", action="store_true", help="Start the training and evaluation jobs")
    args = parser.parse_args()
    if min(args.gpus, args.batch_size, args.micro_batch, args.eval_batch, args.epochs, args.patience) < 1:
        parser.error("Batch sizes, epochs, and patience must be positive")
    if args.batch_size % (args.gpus * args.micro_batch):
        parser.error("Global batch size must be divisible by GPU count times per-GPU micro-batch")
    if any(n < 0 for n in args.readout_counts) or len(set(args.readout_counts)) != len(args.readout_counts):
        parser.error("Readout counts must be unique and nonnegative")
    seed_count = 0
    if args.curriculum_seed_run:
        seed_run = args.curriculum_seed_run
        if not args.readout_curriculum or not (seed_run / "checkpoints/best.pt").is_file():
            parser.error("--curriculum-seed-run requires curriculum mode and an existing best.pt")
        config = json.loads((seed_run / "config.json").read_text(encoding="utf-8"))
        seed_count = config["readout_tokens"]
        pooling = "last" if config.get("readout_pool_tail") == 1 else config.get("readout_pooling", "mean")
        if seed_count < 2 or seed_count % 2 or config["experiment"] != "B2" or pooling != args.readout_pooling:
            parser.error("The curriculum seed must use B2, an even token count, and matching pooling")
    first_count = seed_count + 2
    if args.readout_curriculum and args.readout_counts != list(range(first_count, args.readout_counts[-1] + 1, 2)):
        parser.error(f"Curriculum requires counts starting at {first_count}, without gaps")
    if args.baseline_run and (0 in args.readout_counts or not (args.baseline_run / "checkpoints/best.pt").is_file()):
        parser.error("--baseline-run requires an existing best.pt and replaces count 0")

    out = args.run_dir.resolve()
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={args.gpus}", "-m"]
    common = ["--model", str(args.model.resolve()),
              "--data-dir", str(args.data_dir.resolve()), "--workspace", args.workspace,
              "--lr", "2e-5", "--batch-size", str(args.batch_size),
              "--micro-batch", str(args.micro_batch), "--eval-batch", str(args.eval_batch),
              "--seed", "42", "--epochs", str(args.epochs), "--patience", str(args.patience),
              "--eval-every", "15", "--max-text-tokens", str(args.max_text_tokens),
              "--match-3-token-init" if args.match_3_token_init else "--aligned-readout-init"]
    if args.save_training_state:
        common.append("--save-training-state")
    dataset_name = "Weibo21"
    arms = {f"{dataset_name}-mean-{count}": out / f"{dataset_name}-mean-{count}" for count in args.readout_counts}
    commands = {}
    for count, (name, path) in zip(args.readout_counts, arms.items()):
        command = launcher + ["src2.train", *common, "--run-dir", str(path)]
        command += ["--experiment", "B2", "--readout-tokens", str(count)] if count else ["--experiment", "A"]
        if args.readout_curriculum:
            command.append("--readout-curriculum")
            if count > 2:
                parent_run = args.curriculum_seed_run if count == first_count and args.curriculum_seed_run else (
                    path.with_name(f"{dataset_name}-mean-{count - 2}")
                )
                parent = parent_run.resolve() / "checkpoints" / "best.pt"
                command += ["--readout-parent", str(parent)]
        if count and args.gradient_checkpointing:
            command.append("--gradient-checkpointing")
        if count and args.readout_pooling == "geometric":
            command += ["--readout-pooling", "geometric", "--keep-readout-prefix-init"]
        elif count and args.readout_pooling == "last":
            command += ["--readout-pool-tail", "1"]
        commands[name] = command
    manifest = out / "test-manifest.json"
    commands["heldout"] = launcher + ["src2.test", "--manifest", str(manifest), "--out", str(out / "heldout")]

    test_arms = {count: str(path) for count, path in zip(args.readout_counts, arms.values())}
    if args.curriculum_seed_run:
        test_arms[seed_count] = str(args.curriculum_seed_run.resolve())
    if args.baseline_run:
        test_arms[0] = str(args.baseline_run.resolve())
    for path in args.test_run:
        if not (path / "checkpoints/best.pt").is_file():
            parser.error(f"Missing best.pt for --test-run {path}")
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        count = config["readout_tokens"]
        pooling = "last" if config.get("readout_pool_tail") == 1 else config.get("readout_pooling", "mean")
        if (count in test_arms or pooling != args.readout_pooling or config["dataset"] != "weibo21"
                or config["model"] != str(args.model.resolve()) or config["data_dir"] != str(args.data_dir.resolve())):
            parser.error(f"Duplicate count or incompatible --test-run {path}")
        test_arms[count] = str(path.resolve())

    for command in commands.values():
        print(shlex.join(command), flush=True)
    if not args.execute:
        return

    out.mkdir(parents=True, exist_ok=False)
    (out / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    manifest.write_text(json.dumps({f"{dataset_name}-mean-{n}": test_arms[n] for n in sorted(test_arms)},
                                   indent=2), encoding="utf-8")
    for name, command in commands.items():
        with (out / f"{name}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
