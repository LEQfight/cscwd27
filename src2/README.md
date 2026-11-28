# Weibo21 readout scaling

冻结 Qwen3-0.6B，训练 readout 向量和两层 MLP。A0（0 个 readout）读取正文末 token，B2 读取末层 readout hidden state。B2 支持平均池化、末 token，以及从后向前 `1/2, 1/4, 1/8, ...` 的几何加权。

课程学习每组从上一长度组验证 Macro-F1 最佳的 checkpoint 复制 readout，再追加两个零向量。分类头和 AdamW 每组重新初始化，全部 readout 继续更新。

## 环境与数据

使用支持 BF16 的 CUDA GPU。安装 PyTorch（对应的 CUDA 版本）、Transformers（支持 Qwen3）、pandas、scikit-learn、matplotlib 和 SwanLab。训练需要本地 Qwen3-0.6B 权重及已有的 SwanLab `readout_token` 项目。SwanLab 凭据通过环境变量 `SWANLAB_API_KEY` 提供。

`data/weibo21` 包含 MDFEND-Weibo21 发布的三个划分。加载时排除类别为「无法确定」的记录，保留 0=Real、1=Fake。测试使用官方 1,830 条样本。

## 训练与测试

从仓库根目录执行。下面先训练 A0 和 mean 的 2-token 起点，再将 4–64 tokens 串联训练。省略 `--execute` 可预览批量命令。以 `--gpus 1` 可单卡启动，全局 batch 仍为 64。

```bash
MODEL=/path/to/Qwen3-0.6B
COMMON=(--model "$MODEL" --data-dir data/weibo21 --workspace "$SWANLAB_WORKSPACE" \
        --gpus 2 --match-3-token-init --save-training-state)

python -m src2.scaling "${COMMON[@]}" --run-dir runs/mean-seed \
  --readout-pooling mean --readout-counts 0 2 --epochs 20 --execute

python -m src2.scaling "${COMMON[@]}" --run-dir runs/mean-curriculum \
  --readout-pooling mean --readout-curriculum --readout-counts $(seq 4 2 64) \
  --baseline-run runs/mean-seed/Weibo21-mean-0 \
  --curriculum-seed-run runs/mean-seed/Weibo21-mean-2 --execute

python -m src2.plot_density --scores runs/mean-curriculum/heldout/test-results.json \
  --out runs/mean-curriculum/figures --pooling mean
```

运行累乘权重或末 token 时，将 `--readout-pooling` 设为 `geometric` 或 `last`，使用该池化方式单独训练的 2-token 起点（40 轮上限），并复用 A0。扩展到 128 时可从 64-token run 继续，设置 `--readout-counts $(seq 66 2 128)`，用重复的 `--test-run` 加回已完成的组，绘图时传 `--max-readouts 128`。

默认全局 batch 64，每卡 micro-batch 4、评估 batch 64，seed 42、学习率 `2e-5`。使用 AdamW、2 轮 warmup 和 ReduceLROnPlateau。每 15 个更新步评估训练集和验证集，按轮末验证 Macro-F1 选择最佳 checkpoint，课程学习组最多训练 40 轮、早停 patience 4。批量训练结束后自动测试并输出 Accuracy、AUROC、Macro-F1、Macro-Recall 和 Macro-Precision。

加 `--save-training-state` 保存 `best.pt` 和 `last.pt` 的 AdamW、调度器、轮次和随机数状态。单组续训时复用原 `src2.train` 参数及卡数，换一个 `--run-dir`，增大总 `--epochs`，并加 `--resume-from /path/to/checkpoints/last.pt`。已早停的组需要加 `--reset-early-stop` 才能重新开始计数。
