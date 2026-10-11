"""解码层：一层 = 一个 token 混合器（全注意力或线性注意力）+ 一个前馈网络，各带一个前置 RMSNorm 和一条残差。"""
import torch.nn as nn

from qwen.attention import Attention
from qwen.deltanet import GatedDeltaNet
from qwen.mlp import MLP
from qwen.norm import RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.kind = kind                                                   # "full_attention" 或 "linear_attention"
        if kind == "full_attention":
            self.self_attn = Attention(cfg)
        else:
            self.linear_attn = GatedDeltaNet(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, state=None, mask=None, kv_mask=None):
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin, state, kv_mask) if self.kind == "full_attention" else self.linear_attn(h, state, mask)
        x = x + h                                                          # 残差：混合器只往残差流里"加"东西
        return x + self.mlp(self.post_attention_layernorm(x))
