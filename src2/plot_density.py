"""绘制 Weibo21 上步长为 2 的 readout scaling 曲线。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


METRICS = {
    "acc": "Accuracy", "auroc": "AUROC", "macro_f1": "Macro-F1",
    "macro_recall": "Macro-Recall", "macro_precision": "Macro-Precision",
}
POOLING_LABELS = {"mean": "Mean pooling", "geometric": "Geometric weighting", "last": "Last token"}


def plot_density(rows, output, pooling="mean", max_readouts=64):
    rows = sorted(rows, key=lambda row: row["readout_tokens"])
    counts = tuple(range(0, max_readouts + 1, 2))
    if max_readouts < 2 or max_readouts % 2 or tuple(row["readout_tokens"] for row in rows) != counts:
        raise ValueError(f"Expected unique readout counts from 0 to {max_readouts} in steps of 2")
    if len({row["test_sha256"] for row in rows}) != 1 or any(
        row["dataset"] != "weibo21" or row["test_samples"] != 1830
        or row["model"] != "Qwen3-0.6B" or row["suffix_tokens"] != row["readout_tokens"]
        or row["readout_repeats"] != 1
        or row["readout_pool_tail"] != (1 if pooling == "last" and row["readout_tokens"] else None)
        or row.get("readout_pooling", "mean") != ("geometric" if pooling == "geometric" and row["readout_tokens"] else "mean")
        or row["full_finetune"] or row["lora_rank"] for row in rows
    ):
        raise ValueError("Rows do not share the expected frozen Qwen3/Weibo21 protocol")
    if any(not 0 <= row[metric] <= 1 for metric in METRICS for row in rows):
        raise ValueError("A metric is outside [0, 1]")

    output.mkdir(parents=True, exist_ok=False)
    with (output / "scaling-scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    limits = {row["epochs_cap"] for row in rows if "epochs_cap" in row}
    earlier = [index for index, row in enumerate(rows) if "epochs_cap" in row
               and len(limits) > 1 and row["epochs_cap"] < max(limits)]
    for metric, title in METRICS.items():
        scores = [100 * row[metric] for row in rows]
        top = max(range(len(scores)), key=scores.__getitem__)
        fig, ax = plt.subplots(figsize=(8.4, 4.4), layout="constrained")
        ax.axhline(scores[0], color="#53627A", linestyle=":", label="0-token baseline")
        ax.plot(counts, scores, "-o", color="#0072B2", markersize=3.4, label=POOLING_LABELS[pooling])
        if earlier:
            ax.scatter([counts[i] for i in earlier], [scores[i] for i in earlier],
                       edgecolor="#0072B2", facecolor="white", s=25, zorder=4,
                       label=f"Original {min(limits)}-epoch cap")
        ax.scatter(counts[top], scores[top], color="#D55E00", s=40, zorder=5)
        side = -1 if counts[top] >= max_readouts * .75 else 1
        ax.annotate(f"{counts[top]} tokens · {scores[top]:.2f}%", (counts[top], scores[top]),
                    xytext=(side * 7, 8), textcoords="offset points",
                    ha="right" if side < 0 else "left", color="#A94817", fontsize=9)
        margin = max(0.3, (max(scores) - min(scores)) * 0.13)
        ax.set(title=f"Weibo21 · Qwen3-0.6B · {title}", xlabel="Learnable readout token count",
               ylabel=f"{title} (%)", xticks=range(0, max_readouts + 1, max(2, max_readouts // 8)),
               xlim=(-1.8, max_readouts + 1.8),
               ylim=(max(0, min(scores)-margin), min(100, max(scores)+margin)))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(axis="y", alpha=.22, linewidth=.7)
        ax.legend(loc="best", frameon=False)
        fig.savefig(output / f"{pooling}-{metric}.png", dpi=300, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pooling", choices=tuple(POOLING_LABELS), default="mean")
    parser.add_argument("--max-readouts", type=int, default=64)
    args = parser.parse_args()
    plot_density(json.loads(args.scores.read_text(encoding="utf-8")), args.out, args.pooling, args.max_readouts)


if __name__ == "__main__":
    main()
