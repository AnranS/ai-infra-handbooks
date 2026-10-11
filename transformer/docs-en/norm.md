# 3. RMSNorm: zero-centered normalization

<p class="lead">The input of every layer goes through RMSNorm first, which scales the vector to a "root mean square of 1" and then multiplies each dimension by a learned factor. Qwen3.5's version has a small twist — the stored weight is <code>w − 1</code>, and it is applied as <code>1 + weight</code> — and this chapter explains why. It also writes the "gated RMSNorm" and the L2 normalization that chapter 7 needs. All 54 normalizations in the model are replaced together.</p>

!!! note "What this chapter runs"
    `ch03_norm.py`, about 10 seconds.

@@files@@

## Three normalizations {#三种归一化}

```python title="qwen/norm.py"
"""RMSNorm：把每个向量缩放到"均方根为 1"，再逐维乘一个可学的比例。

Qwen3.5 存的是零中心的权重：文件里的 weight 是 w - 1，用的时候乘 (1 + weight)。
这样初始化为 0 就等于"不缩放"，bf16 下也比存 1.0x 的数精度高（1 附近的 bf16 间隔是 2^-7，0 附近细得多）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))                # zero-centered: 0 means a scale of 1
        self.eps = eps

    def forward(self, x):
        xf = x.float()                                              # statistics are always computed in fp32
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """线性注意力的输出用这个：先归一化，再乘 silu(门)。它的权重不是零中心的（初始为 1）。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * y * F.silu(gate.float())).type_as(x)


def l2norm(x, eps=1e-6):
    """把最后一维缩放到单位长度（线性注意力里 q、k 进记忆表之前用）"""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)
```

How the three differ:

| | Divides by | Then multiplies by | Used where |
| --- | --- | --- | --- |
| `RMSNorm` | the root mean square $\sqrt{\mathrm{mean}(x^2)}$ | $1 + w$ (per dimension) | twice in front of every layer, after the last layer, and on attention's q and k (per head) |
| `GatedRMSNorm` | the root mean square | $w \cdot \mathrm{silu}(z)$ (per dimension; the gate $z$ comes from the input) | the output of linear attention (chapter 7) |
| `l2norm` | the vector length $\sqrt{\sum x^2}$ | nothing | linear attention's q and k (chapter 7) |

Compared with LayerNorm, RMSNorm drops "subtract the mean" and the bias: it only handles scale, not centering, costing half the computation for the same effect. All statistics are computed in fp32: squaring and summing in bf16 loses precision, a convention shared by every implementation.

**Why store w − 1.** At the start of training the scale should be 1. If 1.0 is stored directly, the spacing between neighbouring bf16 numbers near 1 is $2^{-7} \approx 0.008$, and any update smaller than that is swallowed; stored as 0, the spacing near 0 is far finer (bf16 is a floating-point format, denser the closer to 0). So Qwen3.5 stores "how far from 1" as the parameter — hence `zeros` in `__init__` and `1 + weight` in forward. Gemma does the same.

## 54 at once {#54-处一起换}

RMSNorm appears in four kinds of places in the model: every layer's `input_layernorm` and `post_attention_layernorm` (24 × 2), the final `norm` (1), and full attention's `q_norm` / `k_norm` (6 × 2, replaced together with attention in chapter 6). This chapter replaces the first three kinds, 49 in all:

```python title="ch03_norm.py"
"""第 3 章：换掉 24 层 × 2 + 1 处 RMSNorm；看看零中心权重长什么样"""
import os

import torch

from qwen.embedding import Embedding, LMHead
from qwen.harness import load_hf, logits, max_diff, status
from qwen.norm import RMSNorm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

w = W["layers.0.input_layernorm.weight"]
print(f"第 0 层输入归一化的权重：均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}] —— 存的是 w-1，所以在 0 附近")
w = W["norm.weight"]
print(f"最后一层的归一化权重：  均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}] —— 它把尺度调到嵌入表的尺度")

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # replaced in chapter 2
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")                  # this chapter: the final one + two per layer
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
print(f"换掉 49 处 RMSNorm 后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

torch.manual_seed(0)
x = torch.randn(2, 3, cfg.hidden_size) * 5                        # after normalization every vector has an RMS of 1 (with zero weights)
y = RMSNorm(cfg.hidden_size)(x)
print("归一化前每个向量的均方根：", [f"{v:.2f}" for v in x.pow(2).mean(-1).sqrt().flatten()[:3]],
      "归一化后：", [f"{v:.2f}" for v in y.pow(2).mean(-1).sqrt().flatten()[:3]])
```

```text title="output"
第 0 层输入归一化的权重：均值 +0.238，范围 [-0.11, +1.00] —— 存的是 w-1，所以在 0 附近
最后一层的归一化权重：  均值 +3.309，范围 [-0.78, +5.31] —— 它把尺度调到嵌入表的尺度
换掉 49 处 RMSNorm 后，logits 最大差 0.0e+00
已替换 5/9：embed_tokens✓  input_layernorm✓  post_attention_layernorm✓  mlp✗  rotary_emb✗  self_attn✗  linear_attn✗  norm✓  lm_head✓
归一化前每个向量的均方根： ['5.14', '5.05', '4.84'] 归一化后： ['1.00', '1.00', '1.00']
```

!!! interview "How to explain it"
    To explain RMSNorm, first say how it differs from LayerNorm (no mean subtraction, no bias, scale only), then two engineering details: the statistics are computed in fp32; Qwen3.5 / Gemma store $w-1$ because bf16 is finer near 0 than near 1. If you can also say why Pre-Norm (normalization **before** each sublayer) trains better than Post-Norm — the residual stream stays clear all the way, gradients do not have to pass through the normalization — the picture is complete.

## Summary {#小结}

- [x] RMSNorm = divide by the root mean square, then scale per dimension; Qwen3.5's weights are zero-centered (multiply by $1+w$), initialized to 0.
- [x] `GatedRMSNorm` (normalize, then multiply by a silu gate) and `l2norm` are parts for chapter 7's linear attention, placed here for now.
- [x] After replacing 49 of them, the maximum logit difference is 0; 5 of the skeleton's 9 slots are ours.

**File created in this chapter**: `qwen/norm.py`.
