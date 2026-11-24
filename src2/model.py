"""Qwen3 正文分类，读取最后一层的 value 投影。"""

from contextlib import ExitStack

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


READOUT_TOKENS = 3


class ValueReadout(nn.Module):
    readout_tokens = READOUT_TOKENS

    def __init__(self, backbone, *, gradient_checkpointing=False, feature="value", last_only=False):
        super().__init__()
        if backbone.config.model_type != "qwen3":
            raise ValueError("ValueReadout requires a Qwen3 backbone")
        if feature not in ("value", "hidden", "hidden-mean", "value-mean"):
            raise ValueError("Unknown readout feature")
        self.backbone = backbone.requires_grad_(False).eval()
        self.gradient_checkpointing = gradient_checkpointing
        self.feature = feature
        self.readout_positions = 1 if last_only else READOUT_TOKENS
        feature_size = backbone.config.hidden_size
        depth = len(backbone.layers)
        count = min(9, depth) if feature.endswith("-mean") else 1
        self.readout_layers = tuple(round(index * depth / count) for index in range(1, count + 1))
        if feature.startswith("value"):
            # V 只依赖最后一层的输入归一化和逐位置投影。
            last_layer = self.backbone.layers[-1]
            self.last_input_norm = last_layer.input_layernorm
            self.value_projection = last_layer.self_attn.v_proj
            self.value_size = feature_size = self.value_projection.out_features
            self.backbone.layers = self.backbone.layers[:-1]
            self.backbone.norm = nn.Identity()

        # 三个可学习向量的初值是词嵌入均值加独立的小噪声。
        with torch.no_grad():
            mean = backbone.get_input_embeddings().weight.mean(dim=0, dtype=torch.float32)
            vectors = mean.repeat(READOUT_TOKENS, 1)
            vectors.add_(torch.randn_like(vectors) * 0.01)
        self.readout = nn.Parameter(vectors)
        self.classifier = nn.Linear(feature_size, 2)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _read_features(self, embeddings, mask, positions):
        intermediate = []

        def save_readouts(module, args, layer_output):
            states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            # 只保留目标位置，梯度仍可回到 readout 参数。
            intermediate.append(states[:, -self.readout_positions:, :].to(dtype=torch.float32, copy=True))

        with ExitStack() as hooks:
            for layer in self.readout_layers[:-1]:
                block = self.backbone.layers[layer - 1]
                module = block.self_attn.v_proj if self.feature == "value-mean" else block
                handle = module.register_forward_hook(save_readouts)
                hooks.callback(handle.remove)
            output = self.backbone(
                inputs_embeds=embeddings, attention_mask=mask, position_ids=positions,
                use_cache=False, output_hidden_states=False, return_dict=True,
            )
        # Hidden state 来自最终归一化后；value 使用末块自己的输入归一化。
        readout_states = output.last_hidden_state[:, -self.readout_positions:, :].contiguous()
        if self.feature.startswith("value"):
            readout_states = self.value_projection(self.last_input_norm(readout_states))
        if intermediate:
            return torch.stack([*intermediate, readout_states.float()]).mean(dim=0)
        return readout_states

    def forward(self, input_ids, attention_mask, labels=None):
        # 左侧 padding，正文和可学习向量在有效序列中连续排列。
        body = self.backbone.get_input_embeddings()(input_ids)
        suffix = self.readout.to(body.dtype).unsqueeze(0).expand(body.size(0), -1, -1)
        embeddings = torch.cat((body, suffix), dim=1)
        mask = torch.cat((attention_mask, attention_mask.new_ones(body.size(0), READOUT_TOKENS)), dim=1)
        positions = (mask.long().cumsum(dim=1) - 1).masked_fill(mask == 0, 0)

        if self.gradient_checkpointing and self.training and embeddings.requires_grad:
            readouts = checkpoint(self._read_features, embeddings, mask, positions, use_reentrant=False)
        else:
            readouts = self._read_features(embeddings, mask, positions)

        # 将参与分类的位置平均，再对每个样本计算一次交叉熵。
        features = readouts.float().mean(dim=1)
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        key = "values" if self.feature.startswith("value") else "hidden_states"
        return {"logits": logits, "loss": loss, key: readouts}

    def adapter_state(self):
        return {"readout": self.readout.detach().cpu(),
                "classifier": {key: value.detach().cpu()
                               for key, value in self.classifier.state_dict().items()}}

    @torch.no_grad()
    def load_adapter(self, adapter):
        if adapter["readout"].shape != self.readout.shape:
            raise ValueError("Checkpoint readout shape does not match")
        self.classifier.load_state_dict(adapter["classifier"])
        self.readout.copy_(adapter["readout"])


class LastTokenBaseline(nn.Module):
    """原 A 组的正文末 token 和两层分类头。"""

    readout_tokens = 0

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        width = backbone.config.hidden_size
        self.classifier = nn.Sequential(nn.Linear(width, width // 4), nn.GELU(),
                                        nn.Linear(width // 4, 2))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            positions = (attention_mask.long().cumsum(dim=1) - 1).masked_fill(attention_mask == 0, 0)
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                   position_ids=positions, use_cache=False,
                                   output_hidden_states=False, return_dict=True)
            indices = torch.arange(input_ids.size(1), device=input_ids.device)
            last = indices.masked_fill(attention_mask == 0, -1).amax(dim=1)
            rows = torch.arange(input_ids.size(0), device=input_ids.device)
            features = output.last_hidden_state[rows, last].float()
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    def adapter_state(self):
        return {"classifier": {key: value.detach().cpu()
                               for key, value in self.classifier.state_dict().items()}}

    def load_adapter(self, adapter):
        self.classifier.load_state_dict(adapter["classifier"])
