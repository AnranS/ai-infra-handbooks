"""Transformer 语言模型。结构（测试会检查参数量与因果性）：

- 词嵌入 (vocab_size, d_model)；
- n_layers 个 pre-norm 块：x = x + Attn(RMSNorm(x))；x = x + FFN(RMSNorm(x))；
  - RMSNorm：带可学习的权重，eps = 1e-5；
  - Attn：多头因果自注意力，n_heads 个头，head_dim = d_model // n_heads；Q、K、V、O 四个投影都是
    d_model × d_model、无偏置；Q、K 施加 RoPE（θ = rope_theta，相邻两维一组旋转）；
  - FFN：SwiGLU，W2(SiLU(W1 x) * W3 x)，W1、W3 为 d_model → d_ff，W2 为 d_ff → d_model，均无偏置；
- 最后一个 RMSNorm，输出层 d_model → vocab_size（无偏置，不与词嵌入共享）。
"""

from dataclasses import dataclass

import torch


@dataclass
class Config:
    vocab_size: int
    context_length: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384
    rope_theta: float = 10000.0


class TransformerLM(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        raise NotImplementedError

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [batch, seq]（seq ≤ context_length）→ logits: [batch, seq, vocab_size]"""
        raise NotImplementedError
