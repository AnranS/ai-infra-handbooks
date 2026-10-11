# 6. Gated attention

<p class="lead">Only 6 of the 24 layers are full attention, but they are the model's only means of "looking back precisely" — a linear-attention memory table blurs, an attention matrix does not. Compared with the textbook version, Qwen3.5's attention differs in four places: GQA (8 query heads share 2 key/value groups), QK-Norm (q and k each pass through an RMSNorm before being rotated), RoPE on only the first quarter of the dimensions, and an <strong>output gate</strong> — <code>q_proj</code> produces two copies at once, one used as the query and the other passed through a sigmoid and multiplied onto the output. This chapter writes it and replaces the 6 layers.</p>

!!! note "What this chapter runs"
    `ch06_attention.py`, about 10 seconds.

## One forward pass {#一次前向}

```python title="qwen/attention.py"
"""门控注意力：Qwen3.5 的全注意力层，24 层里的 6 层。

和教科书的多头注意力比有四处不同：GQA（8 个 query 头共用 2 组 key/value）、QK-Norm（q 和 k 各过一个
RMSNorm 再旋转）、只旋转前 1/4 维的 RoPE、以及输出门——q_proj 一次算出两份，一份当 query，
一份过 sigmoid 当门，逐维乘在注意力的输出上。
"""
import math

import torch
import torch.nn as nn

from qwen.norm import RMSNorm
from qwen.rope import apply_rope


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, self.n_head, self.n_kv, self.d = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(H, self.n_head * self.d * 2, bias=False)   # twice as wide: half query, half gate
        self.k_proj = nn.Linear(H, self.n_kv * self.d, bias=False)
        self.v_proj = nn.Linear(H, self.n_kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.d, H, bias=False)
        self.q_norm = RMSNorm(self.d, cfg.rms_norm_eps)                     # normalized per head: only head_dim weights
        self.k_norm = RMSNorm(self.d, cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, gate = self.q_proj(x).view(B, T, self.n_head, 2 * self.d).chunk(2, dim=-1)   # each head's 512 dims: the first 256 are the query, the last 256 the gate
        k = self.k_proj(x).view(B, T, self.n_kv, self.d)
        v = self.v_proj(x).view(B, T, self.n_kv, self.d)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                                  # [B, heads, T, d]
        rep = self.n_head // self.n_kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)           # GQA: each KV group is copied to 4 query heads
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d)                               # [B, heads, T, T]
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)               # upper triangle: later tokens are invisible
        scores = scores.masked_fill(causal, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v                                            # [B, heads, T, d]
        out = out.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate.reshape(B, T, -1))   # output gate
        return self.o_proj(out)
```

Read it line by line against the shapes:

| Step | Shape | Note |
| --- | --- | --- |
| `q_proj(x)` | `[B, T, 4096]` → `[B, T, 8, 512]` | 512 dims per head: the first 256 are the query, the last 256 the gate |
| `k_proj`, `v_proj` | `[B, T, 2, 256]` | only 2 key/value groups (GQA) |
| `q_norm`, `k_norm` | unchanged | RMSNorm per head, with only 256 weights; then RoPE |
| `repeat_interleave` | `[B, 8, T, 256]` | each KV group is copied to 4 query heads |
| `scores` | `[B, 8, T, T]` | $qk^\top/\sqrt{256}$, the upper triangle filled with $-\infty$ |
| `softmax @ v` | `[B, 8, T, 256]` → `[B, T, 2048]` | normalize per row, then a weighted sum |
| `* sigmoid(gate)` | `[B, T, 2048]` | the output gate: per dimension, how much of this head's output gets through |
| `o_proj` | `[B, T, 1024]` | back to the width of the residual stream |

**GQA saves KV cache.** 8 query heads, 2 KV groups: the keys/values stored per token drop from 8 × 256 × 2 to 2 × 256 × 2 numbers, a 4× difference in inference memory and bandwidth (chapter 10 does the arithmetic). The quality loss is small, because several query heads looking at the same keys were redundant anyway.

**QK-Norm prevents exploding logits.** During training the scale of $q$ and $k$ drifts; once the dot products get large, softmax degenerates into a one-hot and the gradient vanishes. Normalizing $q$ and $k$ separately before RoPE pins the scale of the dot product. Qwen3 introduced it, Qwen3.5 keeps it.

**The output gate is new in this generation.** It comes from the "Gated Attention" paper behind Qwen3-Next: multiply the attention output by a sigmoid gate computed from the input. The cost is a doubled `q_proj` (12.6M of the 25M parameters are the gate); the benefit is suppressing the "attention sink" (the first token swallowing a large share of attention) and more stable training.

## Replace the 6 layers {#换掉-6-层}

```python title="ch06_attention.py"
"""第 6 章：换掉 6 层全注意力；看一眼注意力矩阵和输出门"""
import os

import torch

from qwen.attention import Attention
from qwen.embedding import Embedding, LMHead
from qwen.harness import AttentionAdapter, RotaryAdapter, load_hf, logits, max_diff, status
from qwen.mlp import MLP
from qwen.norm import RMSNorm
from qwen.rope import rope_tables
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # replaced in chapters 2–5
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
hf.model.rotary_emb = RotaryAdapter(cfg, rope_tables)
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")
    if cfg.layer_types[i] == "full_attention":                                                 # this chapter: 6 layers
        layer.self_attn = AttentionAdapter(load(Attention(cfg), W, f"layers.{i}.self_attn."))
print(f"换掉 6 层全注意力后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

attn = hf.model.layers[3].self_attn.mine                           # layer 3 is the first full-attention layer
n = sum(p.numel() for p in attn.parameters())
print(f"一层注意力 {n / 1e6:.1f}M 参数，其中 q_proj {attn.q_proj.weight.numel() / 1e6:.1f}M（一半是门）")

torch.manual_seed(0)
with torch.no_grad():                                              # take a look at the attention matrix and the gate
    x = torch.randn(1, ids.shape[1], cfg.hidden_size)
    cos, sin = rope_tables(cfg, torch.arange(ids.shape[1]))
    T = ids.shape[1]
    q, gate = attn.q_proj(x).view(1, T, attn.n_head, 2 * attn.d).chunk(2, dim=-1)
    g = torch.sigmoid(gate)
    print(f"输出门 sigmoid(gate)：均值 {g.mean():.2f}，小于 0.1 的比例 {(g < 0.1).float().mean():.0%}，大于 0.9 的比例 {(g > 0.9).float().mean():.0%}")
    causal = torch.ones(T, T, dtype=torch.bool).triu(1)
    print("因果掩码（1 = 看不到）：")
    for row in causal.int()[:4]:
        print("   ", " ".join(str(v) for v in row.tolist()))
```

```text title="output"
换掉 6 层全注意力后，logits 最大差 7.9e-06
已替换 8/9：embed_tokens✓  input_layernorm✓  post_attention_layernorm✓  mlp✓  rotary_emb✓  self_attn✓  linear_attn✗  norm✓  lm_head✓
一层注意力 7.3M 参数，其中 q_proj 4.2M（一半是门）
输出门 sigmoid(gate)：均值 0.50，小于 0.1 的比例 0%，大于 0.9 的比例 1%
因果掩码（1 = 看不到）：
    0 1 1 1 1 1
    0 0 1 1 1 1
    0 0 0 1 1 1
    0 0 0 0 1 1
```

The error went from 0 to the 1e-5 range: our matrix multiplications are blocked differently from HF's, and floating-point addition is not associative. This is normal — every "operator we wrote ourselves" from here on shows a difference of this size; a real bug would be 1e-1 or more.

!!! interview "How to explain it"
    To explain an attention implementation, walking through the shapes is clearest: project → split heads → normalize → rotate → score → mask → softmax → weighted sum → merge heads → gate → project. Then name what each of Qwen3.5's four changes solves: GQA saves KV cache, QK-Norm pins the scale of the dot product, partial rotation keeps some position-free dimensions, the output gate suppresses the attention sink. Leave "why only 6 of 24 layers are attention" to the next chapter: the other 18 use a memory table that does not grow.

## Summary {#小结}

- [x] Gated attention = GQA + QK-Norm + partial RoPE + an output gate; `q_proj` is twice as wide.
- [x] After replacing the 6 layers the maximum logit difference is in the 1e-5 range (a normal floating-point difference from a different multiplication order); 8 of the skeleton's 9 slots are ours.

**File created in this chapter**: `qwen/attention.py` (chapter 10 adds the state and mask arguments).
