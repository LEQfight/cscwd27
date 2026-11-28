"""冻结主模型，读取正文末位置或可学习后缀的表示。"""

from functools import wraps

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


READOUT_TOKENS = 3


def build_classifier(width):
    return nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 2))


def checkpoint_block(forward):
    @wraps(forward)
    def run(*args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)
        return forward(*args, **kwargs)
    return run


class HiddenReadout(nn.Module):
    """将可训练向量接到正文后面，只读取最后一层隐藏状态。"""

    def __init__(self, backbone, *, gradient_checkpointing=False,
                 readout_tokens=READOUT_TOKENS, match_3_token_init=False, pooling="mean",
                 pool_tail=None, aligned_readout_init=False, keep_readout_prefix_init=False):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.readout_tokens = self.suffix_tokens = readout_tokens
        self.readout_positions = pool_tail or self.suffix_tokens
        self.readout_layers = (len(backbone.layers),)
        weights = None
        if pooling == "geometric":
            weights = torch.exp2(-torch.arange(self.readout_positions, 0, -1, dtype=torch.float32))
        self.register_buffer("pool_weights", weights, persistent=False)
        feature_size = backbone.config.hidden_size
        if gradient_checkpointing:
            for block in backbone.layers:
                block.forward = checkpoint_block(block.forward)

        # 默认与 A 同样先初始化分类头。旧三 token 对照则先初始化三个向量。
        if not match_3_token_init:
            self.classifier = build_classifier(feature_size)
        with torch.no_grad():
            mean = backbone.get_input_embeddings().weight.mean(dim=0, dtype=torch.float32)
            initial_count = READOUT_TOKENS if match_3_token_init else readout_tokens
            if aligned_readout_init:
                vectors = torch.stack([mean + torch.randn_like(mean) * 0.01 for _ in range(initial_count)])
            else:
                vectors = mean.repeat(initial_count, 1)
                vectors.add_(torch.randn_like(vectors) * 0.01)
        if match_3_token_init:
            self.classifier = build_classifier(feature_size)
            align_tail = pooling == "geometric" and not keep_readout_prefix_init
            extra = [mean + torch.randn_like(mean) * 0.01
                     for _ in range(readout_tokens - READOUT_TOKENS)]
            if extra:
                if align_tail:
                    # 让不同长度共享同一尾部初始化，对齐权重最大的几个位置。
                    vectors = torch.cat((torch.stack(extra[::-1]), vectors))
                else:
                    vectors = torch.cat((vectors, torch.stack(extra)))
            elif readout_tokens < READOUT_TOKENS:
                # 仍先生成三个初值，再对齐旧分类头的随机数顺序。
                vectors = vectors[-readout_tokens:] if align_tail else vectors[:readout_tokens]
        self.readout = nn.Parameter(vectors)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        # 左侧 padding，正文和可学习向量在有效序列中连续排列。
        body = self.backbone.get_input_embeddings()(input_ids)
        readout = self.readout.to(body.dtype)
        suffix = readout.unsqueeze(0).expand(body.size(0), -1, -1)
        embeddings = torch.cat((body, suffix), dim=1)
        mask = torch.cat((attention_mask, attention_mask.new_ones(body.size(0), self.suffix_tokens)), dim=1)
        positions = (mask.long().cumsum(dim=1) - 1).masked_fill(mask == 0, 0)

        output = self.backbone(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions,
                               use_cache=False, output_hidden_states=False, return_dict=True)
        readouts = output.last_hidden_state[:, -self.readout_positions:, :].contiguous()

        if self.pool_weights is None:
            features = readouts.float().mean(dim=1)
        else:
            features = (readouts.float() * self.pool_weights[:, None]).sum(dim=1)
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    def adapter_state(self):
        return {"readout": self.readout.detach().cpu(),
                "classifier": {key: value.detach().cpu()
                               for key, value in self.classifier.state_dict().items()}}

    @torch.no_grad()
    def initialize_readout_curriculum(self, parent=None):
        # 只继承旧 readout。分类头沿用本组的初始化，新增两个位置从零开始。
        prefix = self.readout[:-2]
        if parent is None:
            if prefix.numel():
                raise ValueError("A readout prefix is required above two tokens")
        else:
            if parent.shape != prefix.shape:
                raise ValueError("Parent readout must have exactly two fewer tokens and the same width")
            prefix.copy_(parent)
        self.readout[-2:].zero_()

    @torch.no_grad()
    def load_adapter(self, adapter):
        if adapter["readout"].shape != self.readout.shape:
            raise ValueError("Checkpoint readout shape does not match")
        self.classifier.load_state_dict(adapter["classifier"])
        self.readout.copy_(adapter["readout"])


class LastTokenBaseline(nn.Module):
    """A0 使用正文最后一个有效位置及相同的 MLP。"""

    readout_tokens = 0
    suffix_tokens = 0

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.classifier = build_classifier(backbone.config.hidden_size)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            positions = (attention_mask.long().cumsum(dim=1) - 1).masked_fill(attention_mask == 0, 0)
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                   position_ids=positions, use_cache=False, return_dict=True)
            # 左侧 padding 使最后一个有效位置位于最后一列。
            features = output.last_hidden_state[:, -1].float()
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    def adapter_state(self):
        return {"classifier": {name: value.detach().cpu()
                               for name, value in self.classifier.state_dict().items()}}

    def load_adapter(self, adapter):
        self.classifier.load_state_dict(adapter["classifier"])
