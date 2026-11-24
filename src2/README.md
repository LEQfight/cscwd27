# Qwen3-0.6B 正文读出实验

所有实验冻结 Qwen3-0.6B，只输入新闻正文。A 使用正文末 token 和两层 MLP。B1–B5 在正文后追加三个共享的可学习向量，使用相同的线性分类头和二分类交叉熵。

| 实验 | 读出特征 | 层 | 分类头直接读取的位置 | 池化 |
|---|---|---|---|---|
| A | Hidden state | 28 | 最后一个有效正文 token | 无 |
| B1 | 原始 value | 28 | 三个 readout token | 位置平均 |
| B2 | Hidden state | 28 | 三个 readout token | 位置平均 |
| B3 | Hidden state | 3、6、9、12、16、19、22、25、28 | 三个 readout token | 层和位置平均 |
| B4 | 原始 value | 同 B3 | 最后一个 readout token | 层平均 |
| B5 | Hidden state | 同 B3 | 最后一个 readout token | 层平均 |

这里的 B1 对应本系列原日志中的 `value-readout`，代码同时接受两个名称。早期另有一项也叫 B1 的 0.5B 正文加推理实验，属于旧方法系列。B4、B5 的前两个 readout 仍参与因果注意力。每层 value 取自 `v_proj`，按顺序拼接 8 个 KV 头，每头 128 维。隐藏状态的末层使用最终归一化输出。A 的分类头为 `1024 → 256 → 2`，其余组为 `1024 → 2`，详见 [model.py](model.py)。

## 数据与训练

[data/weibo21](../data/weibo21) 保留原作者发布的 `train.pkl`、`val.pkl`、`test.pkl` 和原始标签（0=Real、1=Fake）。训练时跳过领域为「无法确定」的记录，保留 5,473 条训练样本和 1,825 条验证样本。训练入口只读取训练和验证集。正文的分词长度上限为 1,533。 [Weibo21 原始发布仓库](https://github.com/kennqiang/MDFEND-Weibo21)

默认 seed 42，目标学习率 `2e-5`，有效 batch size 16，micro-batch 4，评估 batch 16。AdamW 的 weight decay 为 `1e-5`，前两轮线性 warmup，随后按验证 Macro-F1 调整学习率并早停，最多训练 20 轮。每 15 个更新步记录完整训练和验证划分的四项指标，轮末验证选最佳 checkpoint。

从仓库根目录运行。环境需支持 BF16 的 CUDA、PyTorch、Transformers（已用 4.51.3）、pandas、scikit-learn 和 SwanLab。准备本地 Qwen3-0.6B 权重，以及有 `readout_token` 项目的 SwanLab 工作空间。

```bash
python -m src2.train \
  --experiment B1 \
  --model /path/to/Qwen3-0.6B \
  --data-dir data/weibo21 \
  --run-dir runs/B1-seed42 \
  --workspace YOUR_SWANLAB_WORKSPACE
```

每组用独立输出目录。指标写入 `metrics/history.jsonl`，最佳适配器写入 `checkpoints/best.pt`，最终验证结果写入 `result.json`。
