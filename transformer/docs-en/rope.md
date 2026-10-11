# 5. RoPE: rotary position embedding

<p class="lead">Attention by itself does not know the order of tokens — shuffle a sentence and the dot product of every (query, key) pair stays the same. RoPE encodes "which position" as an angle and <strong>rotates the query and the key by that angle</strong>, so that their dot product depends only on their distance. Qwen3.5's version has three unusual details: it rotates only the first quarter of each head, the base frequency is 10,000,000, and the positions have three axes for images and video (M-RoPE). This chapter writes it and replaces HF's <code>rotary_emb</code>.</p>

!!! note "What this chapter runs"
    `ch05_rope.py`, about 10 seconds.

## Turn by an angle {#转一个角}

```python title="qwen/rope.py"
"""旋转位置编码（RoPE）：把"第几个位置"转进 query 和 key 里，让注意力分数只跟相对距离有关。

Qwen3.5 的三个细节：只旋转每个头前 1/4 的维度（256 维里的 64 维，partial_rotary_factor=0.25）；
基频 theta = 10,000,000；为了图像和视频，位置有三个轴（时间、高、宽，M-RoPE），32 个频率按 11/11/10 交错分给三个轴。
纯文本时三个轴的位置相同，M-RoPE 就退化成普通的 RoPE——这个文件写的就是这个退化版。
"""
import torch


def rope_tables(cfg, positions):
    """positions: [T]（或每行不同的 [B, T]）的整数位置 -> cos, sin 各 [T, rot_dim]（rot_dim = head_dim * partial_rotary_factor = 64）"""
    rot = int(cfg.head_dim * cfg.partial_rotary_factor)
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))   # 32 frequencies, from 1 all the way down to 1/theta
    angles = positions.float()[..., None] * inv_freq                                  # [T, 32]: position × frequency = angle
    angles = torch.cat([angles, angles], dim=-1)                                      # [T, 64]: the two halves are paired and share the angle
    return angles.cos(), angles.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    """x: [B, T, 头数, head_dim]。只转前 rot_dim 维，剩下的维度原样拼回去。"""
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    cos, sin = cos[..., None, :].to(x.dtype), sin[..., None, :].to(x.dtype)         # broadcast over batch and heads; the tables are computed in fp32 and cast to x's dtype when used
    x_rot = x_rot * cos + rotate_half(x_rot) * sin                                    # 2-D rotation: (a, b) -> (a cos - b sin, b cos + a sin)
    return torch.cat([x_rot, x_pass], dim=-1)
```

Think of each head's 256 dimensions as 128 pairs $(a, b)$. For the $i$-th pair, position $p$ corresponds to the angle $p \cdot \theta_i$, where $\theta_i = \text{base}^{-2i/d}$ falls from 1 all the way to $1/\text{base}$ — the first pairs turn fast (telling neighbouring positions apart), the last ones turn extremely slowly (telling near from far thousands of positions away). The dot product of two rotated vectors:

$$
\langle R_{p}\,q,\; R_{p'}\,k \rangle = \langle q,\; R_{p'-p}\,k \rangle
$$

depends only on $p' - p$. That is all of RoPE: the absolute position is applied to the vectors, the relative position shows up in the dot product.

**Only a quarter is rotated.** `partial_rotary_factor = 0.25`: of the 256 dimensions only the first 64 (32 pairs) are rotated; the other 192 carry no position information. Qwen3.5 inherits this from Qwen3-Next: keeping some dimensions for "position-independent matching" is more stable over long contexts.

**Base frequency 10,000,000.** The classic value is 10,000; the larger the base, the slower the slowest pairs turn, so that a 262K context does not "wrap around" and confuse near with far.

**M-RoPE.** In multimodal use the position has three axes: time (the order of text, the frames of a video), image height and image width. The 32 frequencies are interleaved across the three axes according to `mrope_section = [11, 11, 10]` (frequencies 0, 3, 6 … go to the time axis, 1, 4, 7 … to height, 2, 5 … to width). For pure text the three axes have the same positions and the three tables are identical, so the code above is written for a single axis — and chapter 1's `RotaryAdapter` taking axis 0 of HF's `[3, B, T]` positions is the same fact.

## Plug it in and check {#换上去对答案}

```python title="ch05_rope.py"
"""第 5 章：自己算 cos/sin 表，和 HF 的逐位对；换掉 rotary_emb；看一眼"相对位置"是怎么出现在点积里的"""
import os

import torch

from qwen.embedding import Embedding, LMHead
from qwen.harness import RotaryAdapter, load_hf, logits, max_diff, status
from qwen.mlp import MLP
from qwen.norm import RMSNorm
from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

pos = torch.arange(ids.shape[1])
cos, sin = rope_tables(cfg, pos)
hf_cos, hf_sin = hf.model.rotary_emb(torch.zeros(1), pos.view(1, 1, -1).expand(3, 1, -1))   # HF wants three-axis positions [3, B, T]
print(f"cos 表 {tuple(cos.shape)}（HF 的是 {tuple(hf_cos.shape)}），和 HF 最大差 {max(max_diff(cos, hf_cos[0]), max_diff(sin, hf_sin[0])):.1e}")
rot = int(cfg.head_dim * cfg.partial_rotary_factor)
print(f"每个头 {cfg.head_dim} 维，只旋转前 {rot} 维（{rot // 2} 对）；最快的一对每个位置转 {1:.2f} 弧度，最慢的一对转 {1 / cfg.rope_theta ** ((rot - 2) / rot):.1e} 弧度")

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # replaced in chapters 2–4
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")
hf.model.rotary_emb = RotaryAdapter(cfg, rope_tables)                                          # this chapter
print(f"换掉 rotary_emb 后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

torch.manual_seed(0)                                              # relative position: the same q and k at different positions but the same distance give the same dot product
q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)
def dot(pq, pk):
    cq, sq = rope_tables(cfg, torch.tensor([pq]))
    ck, sk = rope_tables(cfg, torch.tensor([pk]))
    return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()
print(f"q 在 3、k 在 1：{dot(3, 1):.4f}；q 在 103、k 在 101：{dot(103, 101):.4f}；q 在 3、k 在 2：{dot(3, 2):.4f}")
```

```text title="output"
cos 表 (6, 64)（HF 的是 (1, 6, 64)），和 HF 最大差 0.0e+00
每个头 256 维，只旋转前 64 维（32 对）；最快的一对每个位置转 1.00 弧度，最慢的一对转 1.7e-07 弧度
换掉 rotary_emb 后，logits 最大差 0.0e+00
已替换 7/9：embed_tokens✓  input_layernorm✓  post_attention_layernorm✓  mlp✓  rotary_emb✓  self_attn✗  linear_attn✗  norm✓  lm_head✓
q 在 3、k 在 1：-20.3131；q 在 103、k 在 101：-20.3131；q 在 3、k 在 2：-19.6979
```

The last line is RoPE's "relativity": (3, 1) and (103, 101) give the same dot product because both distances are 2; (3, 2) differs because the distance changed.

!!! interview "How to explain it"
    To explain RoPE, first say what it solves: attention is insensitive to order and needs position information, while additive position encodings leak an "absolute position" term into the dot product. Then the method: rotate by an angle proportional to the position, and only the relative position survives in the dot product. Finally what the two numbers mean: the base decides how long a distance can be resolved (Qwen3.5 pairs 1e7 with a 262K context), and partial rotation leaves some dimensions without position. Mentioning M-RoPE — three position axes, collapsing to one for pure text — shows you have read this generation's multimodal implementation.

## Summary {#小结}

- [x] RoPE encodes position $p$ as the angle $p\theta_i$ and rotates the query and the key; the dot product depends only on the relative distance.
- [x] Qwen3.5: only the first 64 dimensions of each head, base 1e7, three-axis M-RoPE (one axis for pure text); our cos/sin tables equal HF's element by element.
- [x] 7 of the skeleton's 9 slots are ours.

**File created in this chapter**: `qwen/rope.py`.
