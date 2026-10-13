# 6. Gated attention

<p class="lead">Only 6 of the 24 layers are full attention, but they are the model's only means of "looking back precisely" — a linear-attention memory table blurs, an attention matrix does not. Compared with the textbook version, Qwen3.5's attention differs in four places: GQA (8 query heads share 2 key/value groups), QK-Norm (q and k each pass through an RMSNorm before being rotated), RoPE on only the first quarter of the dimensions, and an <strong>output gate</strong> — <code>q_proj</code> produces two copies at once, one used as the query and the other passed through a sigmoid and multiplied onto the output. This chapter writes it, then uses real weights and real input to draw the attention matrix and look at it.</p>

!!! note "What this chapter runs"
    `ch06_attention.py`, about 10 seconds.

@@files@@

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

![Figure: one forward pass of gated attention](assets/figures/gated-attention.svg){.aig-svg}

**GQA saves KV cache.** 8 query heads, 2 KV groups: the keys/values stored per token drop from 8 × 256 × 2 to 2 × 256 × 2 numbers, a 4× difference in inference memory and bandwidth (chapter 10 does the arithmetic). The quality loss is small, because several query heads looking at the same keys were redundant anyway.

**QK-Norm prevents exploding logits.** During training the scale of $q$ and $k$ drifts; once the dot products get large, softmax degenerates into a one-hot and the gradient vanishes. Normalizing $q$ and $k$ separately before RoPE pins the scale of the dot product. Qwen3 introduced it, Qwen3.5 keeps it.

**The output gate is new in this generation.** It comes from the "Gated Attention" paper behind Qwen3-Next: multiply the attention output by a sigmoid gate computed from the input. The cost is a doubled `q_proj` (12.6M of the 25M parameters are the gate); the benefit is suppressing the "attention sink" (the first token swallowing a large share of attention) and more stable training.

## Run it: what the attention matrix looks like {#跑起来注意力矩阵长什么样}

```python title="ch06_attention.py" ci="loose"
"""第 6 章：门控注意力。用真实权重和真实输入跑一层，把注意力矩阵和输出门拿出来看"""
import math
import os

import torch
from tokenizers import Tokenizer

from qwen.attention import Attention
from qwen.embedding import Embedding
from qwen.norm import RMSNorm
from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
attn = load(Attention(cfg), W, "layers.3.self_attn.")                                          # layer 3 is the first full-attention layer
n = sum(p.numel() for p in attn.parameters())
print(f"一层注意力 {n / 1e6:.1f}M 参数，其中 q_proj {attn.q_proj.weight.numel() / 1e6:.1f}M（一半是门）")

tk = Tokenizer.from_file(str(path / "tokenizer.json"))         # build a real input with the parts from chapters 2 and 3
ids = torch.tensor([tk.encode("今天天气不错，我们去公园").ids])
T = ids.shape[1]
h = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.3.input_layernorm.")(load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")(ids))
cos, sin = rope_tables(cfg, torch.arange(T))
y = attn(h, cos, sin)
print(f"输入 {tuple(h.shape)} -> 输出 {tuple(y.shape)}")

with torch.no_grad():                                                                          # 1. compute the attention matrix and look at it: every row sums to 1, the upper triangle is 0
    q, gate = attn.q_proj(h).view(1, T, attn.n_head, 2 * attn.d).chunk(2, dim=-1)
    k = attn.k_proj(h).view(1, T, attn.n_kv, attn.d)
    q, k = apply_rope(attn.q_norm(q), cos, sin), apply_rope(attn.k_norm(k), cos, sin)
    scores = q[0, :, 0] @ k[0, :, 0].T / math.sqrt(attn.d)                                     # query head 0 × its KV group
    probs = torch.softmax(scores.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf")), dim=-1)
    words = [tk.decode([i]) for i in ids[0].tolist()]
    print("第 0 个头的注意力矩阵（行 = query，列 = key）：")
    for w, row in zip(words, probs):
        print(f"  {w:<4}", " ".join(f"{p:.2f}" for p in row.tolist()))
    g = torch.sigmoid(gate)                                                                    # 2. the output gate: how far each dimension of each head is open
    print(f"输出门 sigmoid(gate)：均值 {g.mean():.2f}，小于 0.1 的比例 {(g < 0.1).float().mean():.0%}，大于 0.9 的比例 {(g > 0.9).float().mean():.0%}")
print(f"GQA：{attn.n_head} 个 query 头共用 {attn.n_kv} 组 KV，每个 token 要存的 KV 是 {2 * attn.n_kv * attn.d} 个数，不用 GQA 要存 {2 * attn.n_head * attn.d} 个")
```

```text title="输出"
一层注意力 7.3M 参数，其中 q_proj 4.2M（一半是门）
输入 (1, 6, 1024) -> 输出 (1, 6, 1024)
第 0 个头的注意力矩阵（行 = query，列 = key）：
  今天   1.00 0.00 0.00 0.00 0.00 0.00
  天气   0.07 0.93 0.00 0.00 0.00 0.00
  不错   0.01 0.97 0.02 0.00 0.00 0.00
  ，    0.00 0.23 0.76 0.00 0.00 0.00
  我们去  0.05 0.01 0.00 0.43 0.50 0.00
  公园   0.00 0.00 0.13 0.21 0.06 0.60
输出门 sigmoid(gate)：均值 0.49，小于 0.1 的比例 2%，大于 0.9 的比例 2%
GQA：8 个 query 头共用 2 组 KV，每个 token 要存的 KV 是 1024 个数，不用 GQA 要存 4096 个
```

Every row of the attention matrix sums to 1 and the upper triangle is all 0 (the causal mask): the first token can only see itself, hence 1.00; "不错" (nice) gives 97% of its attention to the previous token "天气" (weather), while "公园" (park) spreads it over "不错", "，", "我们去" (we go to) and itself. This is only head 0 of the first attention layer, and the input has only passed through the embedding and the normalization (not the three linear-attention layers before it), so it looks like it is doing something local such as "look at the previous word" — deeper heads look further. The output gate's mean of 0.49, with very few extreme values, says that after training the gate mostly fine-tunes rather than switching on and off.

!!! interview "How to explain it"
    To explain an attention implementation, walking through the shapes is clearest: project → split heads → normalize → rotate → score → mask → softmax → weighted sum → merge heads → gate → project. Then name what each of Qwen3.5's four changes solves: GQA saves KV cache, QK-Norm pins the scale of the dot product, partial rotation keeps some position-free dimensions, the output gate suppresses the attention sink. Leave "why only 6 of 24 layers are attention" to the next chapter: the other 18 use a memory table that does not grow.

## Summary {#小结}

- [x] Gated attention = GQA + QK-Norm + partial RoPE + an output gate; `q_proj` is twice as wide.
- [x] Every row of the attention matrix sums to 1 and the upper triangle is 0; head 0 of the first layer leans towards the previous token; GQA cuts the KV stored per token from 4096 numbers to 1024.

**File created in this chapter**: `qwen/attention.py` (chapter 10 adds the state and mask arguments).
