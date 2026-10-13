# 5. RoPE: rotary position embedding

<p class="lead">Attention by itself does not know the order of tokens — shuffle a sentence and the dot product of every (query, key) pair stays the same. RoPE encodes "which position" as an angle and <strong>rotates the query and the key by that angle</strong>, so that their dot product depends only on their distance. Qwen3.5's version has three unusual details: it rotates only the first quarter of each head, the base frequency is 10,000,000, and the positions have three axes for images and video (M-RoPE). This chapter writes it and looks at how fast each pair turns and how the relative position shows up in the dot product.</p>

!!! note "What this chapter runs"
    `ch05_rope.py`, about 10 seconds.

@@files@@

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

**M-RoPE.** In multimodal use the position has three axes: time (the order of text, the frames of a video), image height and image width. The 32 frequencies are interleaved across the three axes according to `mrope_section = [11, 11, 10]` (frequencies 0, 3, 6 … go to the time axis, 1, 4, 7 … to height, 2, 5 … to width). For pure text the three axes have the same positions and the three tables are identical, so the code above is written for a single axis (in the Hugging Face implementation the position tensor is `[3, B, T]`, and for pure text the three axes hold the same positions).

## Run it: how fast it turns, relative position, the unrotated dimensions {#跑起来转得多快相对位置没转的维度}

```python title="ch05_rope.py" ci="loose"
"""第 5 章：自己算 cos/sin 表，看每一对转得多快、"相对位置"是怎么出现在点积里的、没旋转的维度原样不动"""
import math
import os

import torch

from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
cfg = Config.load(find_model())
rot = int(cfg.head_dim * cfg.partial_rotary_factor)
cos, sin = rope_tables(cfg, torch.arange(6))
print(f"cos 表 {tuple(cos.shape)}：每个头 {cfg.head_dim} 维只旋转前 {rot} 维（{rot // 2} 对），位置 0 的 cos 全是 1：{cos[0, :4].tolist()}")

inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))   # 1. how fast each pair turns: how many positions for a full turn
for i in (0, 8, 16, 24, 31):
    print(f"  第 {i:2d} 对：每个位置转 {inv_freq[i]:.2e} 弧度，转满一圈要 {2 * math.pi / inv_freq[i]:.3g} 个位置")

torch.manual_seed(0)                                              # 2. relative position: the same q and k at different positions but the same distance give the same dot product
q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)


def dot(pq, pk):
    cq, sq = rope_tables(cfg, torch.tensor([pq]))
    ck, sk = rope_tables(cfg, torch.tensor([pk]))
    return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()


print(f"q 在 3、k 在 1：{dot(3, 1):.4f}；q 在 103、k 在 101：{dot(103, 101):.4f}；q 在 3、k 在 2：{dot(3, 2):.4f}")
print(f"距离从 0 拉到 2000，点积怎么变：", [f"{dot(d, 0):.1f}" for d in (0, 1, 2, 10, 100, 2000)])

x = torch.randn(1, 6, 8, cfg.head_dim)                            # 3. partial rotation: the last 192 dims stay untouched
y = apply_rope(x, cos, sin)
print(f"旋转后前 {rot} 维变了（最大差 {(y[..., :rot] - x[..., :rot]).abs().max():.2f}），后 {cfg.head_dim - rot} 维没动（最大差 {(y[..., rot:] - x[..., rot:]).abs().max():.1e}）")
```

```text title="输出"
cos 表 (6, 64)：每个头 256 维只旋转前 64 维（32 对），位置 0 的 cos 全是 1：[1.0, 1.0, 1.0, 1.0]
  第  0 对：每个位置转 1.00e+00 弧度，转满一圈要 6.28 个位置
  第  8 对：每个位置转 1.78e-02 弧度，转满一圈要 353 个位置
  第 16 对：每个位置转 3.16e-04 弧度，转满一圈要 1.99e+04 个位置
  第 24 对：每个位置转 5.62e-06 弧度，转满一圈要 1.12e+06 个位置
  第 31 对：每个位置转 1.65e-07 弧度，转满一圈要 3.8e+07 个位置
q 在 3、k 在 1：-20.3131；q 在 103、k 在 101：-20.3131；q 在 3、k 在 2：-19.6979
距离从 0 拉到 2000，点积怎么变： ['-18.5', '-19.7', '-20.3', '-18.0', '-19.9', '-6.4']
旋转后前 64 维变了（最大差 5.38），后 192 维没动（最大差 0.0e+00）
```

Pair 0 completes a full turn in 6.28 positions, pair 31 needs 38 million — the fast pairs tell neighbouring positions apart, the slow pairs tell near from far thousands or tens of thousands of positions away, and a 262K context is less than a quarter of a turn in the eyes of the slowest pairs. The middle line is RoPE's "relativity": (3, 1) and (103, 101) give the same dot product because both distances are 2; (3, 2) differs because the distance changed; from distance 0 to 2000 the dot product keeps changing. The last line: the last 192 dimensions are untouched.

!!! interview "How to explain it"
    To explain RoPE, first say what it solves: attention is insensitive to order and needs position information, while additive position encodings leak an "absolute position" term into the dot product. Then the method: rotate by an angle proportional to the position, and only the relative position survives in the dot product. Finally what the two numbers mean: the base decides how long a distance can be resolved (Qwen3.5 pairs 1e7 with a 262K context), and partial rotation leaves some dimensions without position. Mentioning M-RoPE — three position axes, collapsing to one for pure text — shows you have read this generation's multimodal implementation.

## Summary {#小结}

- [x] RoPE encodes position $p$ as the angle $p\theta_i$ and rotates the query and the key; the dot product depends only on the relative distance.
- [x] Qwen3.5: only the first 64 dimensions of each head, base 1e7, three-axis M-RoPE (one axis for pure text); the fastest pair turns once in 6 positions, the slowest in 38 million.

**File created in this chapter**: `qwen/rope.py`.
