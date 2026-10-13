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

## Run it: the scale before and after normalization {#跑起来归一化前后的尺度}

RMSNorm appears in four kinds of places in the model: every layer's `input_layernorm` and `post_attention_layernorm` (24 × 2), the final `norm` (1), and full attention's `q_norm` / `k_norm` (6 × 2, used together with attention in chapter 6); the gated version lives in the 18 linear-attention layers. Run it and look at three things: the root mean square before and after normalization, where the real weights put the scale, and why bf16 needs $w - 1$ to be stored:

```python title="ch03_norm.py"
"""第 3 章：三种归一化。看归一化前后的尺度、真实权重把尺度调到了哪里、为什么要存 w-1"""
import os

import torch

from qwen.norm import GatedRMSNorm, RMSNorm, l2norm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
torch.manual_seed(0)
x = torch.randn(2, 3, cfg.hidden_size) * 5
rms = lambda t: [f"{v:.2f}" for v in t.float().pow(2).mean(-1).sqrt().flatten()[:3]]

print("归一化前每个向量的均方根：", rms(x), "；权重为 0（即比例为 1）时归一化后：", rms(RMSNorm(cfg.hidden_size)(x)))   # 1. what it does
norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.0.input_layernorm.")
final = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
print("第 0 层输入归一化之后：", rms(norm(x)), "；最后一层归一化之后：", rms(final(x)), "（它把尺度放大到嵌入表的尺度）")
w = W["layers.0.input_layernorm.weight"]
print(f"第 0 层输入归一化的权重：均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}] —— 存的是 w-1，所以在 0 附近")
w = W["norm.weight"]
print(f"最后一层的归一化权重：  均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}]")

one = torch.tensor(1.0 + 1e-3)                                              # 2. why w-1 is stored: bf16 is coarse near 1 and fine near 0
print(f"bf16 存 1 + 0.001 得到 {one.bfloat16().item():.4f}（更新被吞掉）；存 0.001 得到 {torch.tensor(1e-3).bfloat16().item():.6f}（留住了）")
print(f"bf16 输入 -> 输出 dtype {RMSNorm(cfg.hidden_size)(x.bfloat16()).dtype}，统计量在 fp32 里算")

gated = load(GatedRMSNorm(cfg.linear_value_head_dim, cfg.rms_norm_eps), W, "layers.0.linear_attn.norm.")   # 3. the gated version: close the gate and the output vanishes
h = torch.randn(4, cfg.linear_value_head_dim)
for g in (-6.0, 0.0, 6.0):
    print(f"  门 = {g:+.0f}：输出的均方根 {gated(h, torch.full_like(h, g)).pow(2).mean(-1).sqrt().mean():.3f}（silu({g:+.0f}) = {torch.nn.functional.silu(torch.tensor(g)):.3f}）")
print("l2norm 之后向量的长度：", [f"{v:.3f}" for v in l2norm(x).norm(dim=-1).flatten()[:3]])
print(f"全模型 {cfg.num_hidden_layers * 2 + 1} 处 RMSNorm、{cfg.layer_types.count('full_attention') * 2} 处按头的 QK-Norm、{cfg.layer_types.count('linear_attention')} 处带门的 RMSNorm，都是这两个类")
```

```text title="输出"
归一化前每个向量的均方根： ['5.14', '5.05', '4.84'] ；权重为 0（即比例为 1）时归一化后： ['1.00', '1.00', '1.00']
第 0 层输入归一化之后： ['1.24', '1.24', '1.25'] ；最后一层归一化之后： ['4.33', '4.32', '4.34'] （它把尺度放大到嵌入表的尺度）
第 0 层输入归一化的权重：均值 +0.238，范围 [-0.11, +1.00] —— 存的是 w-1，所以在 0 附近
最后一层的归一化权重：  均值 +3.309，范围 [-0.78, +5.31]
bf16 存 1 + 0.001 得到 1.0000（更新被吞掉）；存 0.001 得到 0.000999（留住了）
bf16 输入 -> 输出 dtype torch.bfloat16，统计量在 fp32 里算
  门 = -6：输出的均方根 0.014（silu(-6) = -0.015）
  门 = +0：输出的均方根 0.000（silu(+0) = 0.000）
  门 = +6：输出的均方根 5.730（silu(+6) = 5.985）
l2norm 之后向量的长度： ['1.000', '1.000', '1.000']
全模型 49 处 RMSNorm、12 处按头的 QK-Norm、18 处带门的 RMSNorm，都是这两个类
```

With the weight at 0, the root mean square after normalization is exactly 1; layer 0's real weights move it to 1.24 and the last layer's to 4.3 — that is the scale of the embedding table (chapter 2 had $\text{logit} = h \cdot E$, so the scale of $h$ has to match $E$). The bf16 line is the reason for storing $w-1$: 1.001 turns straight back into 1.0 in bf16, while 0.001 survives. The gated version: with the gate at −6 the output is almost 0, at +6 it is amplified 6× — chapter 7's linear attention relies on it to decide how much of each head's output to let through.

!!! interview "How to explain it"
    To explain RMSNorm, first say how it differs from LayerNorm (no mean subtraction, no bias, scale only), then two engineering details: the statistics are computed in fp32; Qwen3.5 / Gemma store $w-1$ because bf16 is finer near 0 than near 1. If you can also say why Pre-Norm (normalization **before** each sublayer) trains better than Post-Norm — the residual stream stays clear all the way, gradients do not have to pass through the normalization — the picture is complete.

## Summary {#小结}

- [x] RMSNorm = divide by the root mean square, then scale per dimension; Qwen3.5's weights are zero-centered (multiply by $1+w$), initialized to 0.
- [x] `GatedRMSNorm` (normalize, then multiply by a silu gate) and `l2norm` are parts for chapter 7's linear attention, placed here for now.
- [x] The real weights move the scale from 1 to 1.24 (layer 0) and 4.3 (the last layer); storing 1.001 in bf16 loses it to 1.0, hence $w-1$.

**File created in this chapter**: `qwen/norm.py`.
