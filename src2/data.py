"""加载 Weibo21 的文本及标签。"""

import pandas as pd
import torch
from torch.utils.data import DataLoader


def load_split(path, tokenizer, max_text_tokens):
    # 读取原作者发布的 pickle，保留原始 0=Real、1=Fake 标签。
    frame = pd.read_pickle(path)
    frame = frame.loc[frame["category"] != "无法确定", ["content", "label"]]
    if frame.empty or not frame["label"].isin((0, 1)).all():
        raise ValueError(f"Missing samples or invalid labels in {path}")
    texts = frame["content"].tolist()
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError(f"Empty or invalid text in {path}")
    ids = tokenizer(texts, add_special_tokens=False, truncation=True,
                    max_length=max_text_tokens, return_attention_mask=False)["input_ids"]
    if not all(ids):
        raise ValueError(f"Tokenizer returned an empty sequence in {path}")
    return list(zip(ids, map(int, frame["label"])))


def collate(rows, pad_id):
    # 每批只填充到该批最长的正文，并用 mask 屏蔽左侧 padding。
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for index, (ids, _) in enumerate(rows):
        input_ids[index, -len(ids):] = torch.tensor(ids)
        mask[index, -len(ids):] = 1
    return {"input_ids": input_ids, "attention_mask": mask,
            "labels": torch.tensor([label for _, label in rows], dtype=torch.long)}


def evaluation_loader(rows, batch_size, collate_fn, rank=0, world_size=1):
    # 保留单卡的长度排序和批边界，交错分配整批，每条样本只评估一次。
    ordered = sorted(rows, key=lambda row: len(row[0]))
    batches = range(0, len(ordered), batch_size)
    local_rows = [row for start in batches[rank::world_size] for row in ordered[start:start + batch_size]]
    return DataLoader(local_rows, batch_size=batch_size, collate_fn=collate_fn)
