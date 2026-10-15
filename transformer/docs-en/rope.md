# 5. RoPE: rotary position embedding

<p class="lead">Attention looks only at content, not at position: shuffle the words of a sentence and not a single (query, key) dot product changes. RoPE's fix is to <strong>turn the query and the key by an angle that depends on their position</strong>: a vector at position p is turned p steps, so the angle between two vectors keeps only their distance. This chapter first turns a 2D vector with concrete numbers, then shows how Qwen3.5 spreads the idea over 64 dimensions and 32 speeds, and finally reads the code line by line.</p>

!!! note "What this chapter runs"
    `ch05_rope.py`, about 10 seconds.

@@files@@

## Why attention needs positions {#为什么注意力需要位置}

In chapter 6's attention, the score token i gives token j is $q_i \cdot k_j$. Both q and k are computed from that token's own vector alone, with nothing about where it sits. So in "the dog bit the man" and "the man bit the dog", the score "man" gives "dog" is exactly the same, and the model cannot tell who bit whom. The causal mask only decides which words each word can see; among the words it can see, which came first does not show up in the scores.

There are two kinds of ways to tell the model about positions:

- **Add a position vector**: prepare a vector for position p and add it to the word vector (the original Transformer, GPT-2). Position and content get mixed, and the dot product picks up terms that depend on absolute positions, while what the model usually cares about is "how many words apart".
- **Turn q and k by an angle**: leave the word vectors alone, and just before scoring, turn q and k by an angle that depends on their position. After the turn, the score depends only on the **difference** between the two positions. This is RoPE, and nearly every large model today uses it.

## 2D first: turning by an angle {#先看两维转一个角}

Turning a 2D vector $(a, b)$ counterclockwise by an angle $\varphi$ gives

$$
(a, b) \;\to\; (a\cos\varphi - b\sin\varphi,\;\; a\sin\varphi + b\cos\varphi)
$$

Its length stays the same; only its direction changes. RoPE has a single rule: **a vector at position p is turned by $p\theta$**, where $\theta$ is how much more it turns per position.

Why does the dot product then see only the distance? A dot product is the product of the two lengths times the cosine of the angle between them. q at position m is turned by $m\theta$ and k at position n by $n\theta$, so the angle between them changes by $(m - n)\theta$, which depends only on the difference $m - n$. With concrete numbers, take $\theta = 30°$ and q = k = $(1, 0)$:

- q at position 3, k at position 1: q turns to 90° and k to 30°, the angle between them is 60°, and the dot product is $\cos 60° = 0.5$.
- Move both back by 100: q at 103 has turned 3090°, which is 210° after removing full turns; k at 101 turns to 150°. The angle is still 60° and the dot product still 0.5.
- q at 3, k at 2: the distance is now 1, the angle 30°, and the dot product $\cos 30° = 0.866$.

![Figure: turning by an angle in 2D: 30° more per position; move q and k by 100 together and the angle and dot product stay the same](assets/figures/rope-2d.svg){.aig-svg}

As a formula, with $R_\varphi$ meaning "turn by $\varphi$":

$$
\langle R_{m\theta}\, q,\;\; R_{n\theta}\, k \rangle \;=\; \langle q,\;\; R_{(n-m)\theta}\, k \rangle
$$

Only $n - m$ is left on the right. Absolute positions go into the vectors, relative positions come out in the dot product: that is the whole idea of RoPE.

## One hand is not enough: 32 pairs, 32 speeds {#一根指针不够32-对32-种转速}

A single $\theta$ has a flaw: after a full turn it is back where it started. With $\theta = 30°$, two words 12 apart are at an angle of 360°, exactly like distance 0, and the model cannot tell "myself" from "the 12th word back".

The fix is the same as on a clock. The second hand goes round once a minute, so from the second hand alone you cannot tell now from a minute ago; add the minute hand and the hour hand, and the hands together pin down the time. RoPE cuts a head's vector into many pairs, and each pair is a "hand" with its own speed:

$$
\theta_i = \text{base}^{-2i/d}, \qquad i = 0, 1, \dots, d/2 - 1
$$

In Qwen3.5, $d = 64$ dimensions of each head take part in the rotation, that is 32 pairs, and the base is 10,000,000:

- Pair 0 turns 1 radian per position, about 57°, a full turn every 6.28 positions: the second hand, telling apart "one or two words away".
- Each pair turns 1.65 times slower than the one before.
- Pair 31 needs 38 million positions for a full turn: the hour hand, telling apart "thousands or tens of thousands of words away".

**Why the base is so large.** The slowest hand must not complete a turn within the whole context, or far distances would again blur into near ones. Qwen3.5's context is 262,144 tokens; with the classic base of 10,000, the slowest pair would complete a turn every 47,000 positions; with 10,000,000 it takes 38 million.

The dials below can be dragged. Each dial is one dimension pair, the blue hand is q and the orange hand is k. Move m and n back together and the angle on every dial stays the same; change only one of them and the fast dials spin while the slow ones barely move.

<div class="aig-widget" data-widget="rope-clock"></div>

## Which two of the 256 dimensions form a pair {#256-维里哪两维是一对}

Each Qwen3.5 head has 256 dimensions, and only the first 64 take part in the rotation (`partial_rotary_factor = 0.25`); the other 192 stay as they are and do position-free matching. Qwen3.5 inherits this from Qwen3-Next; it is more stable over long contexts.

How are the first 64 paired? Not as neighbours (0, 1), (2, 3), …, but **first half with second half**: dimension j pairs with dimension j + 32 and uses speed j. The RoPE paper pairs neighbouring dimensions; the Qwen family and most models in Hugging Face pair the two halves. The two layouts differ only by a fixed permutation of the dimensions and are mathematically equivalent, but the weights must be rotated with the layout they were trained with, or the model's output turns to nonsense.

![Figure: a head's 256 dims: the first 64 rotate, dim j pairs with dim j + 32; the last 192 do not rotate](assets/figures/rope-pairs.svg){.aig-svg}

With the halves paired, "turning by an angle" becomes two element-wise multiplications: swap the two halves of x and put a minus sign on the front half to get `rotate_half(x)`, then compute `x * cos + rotate_half(x) * sin`. With $a$ in dimension j and $b$ in dimension j + 32, dimension j gets $a\cos - b\sin$ and dimension j + 32 gets $b\cos + a\sin$: exactly the 2D rotation above.

**M-RoPE.** In multimodal use the position has three axes: time (the order of text, the frames of a video), image height and image width. The 32 frequencies are interleaved across the three axes according to `mrope_section = [11, 11, 10]`: frequencies 0, 3, 6 … go to the time axis, 1, 4, 7 … to height, 2, 5 … to width. For pure text the three axes hold the same positions and the three tables are identical, so the code below is written for a single axis. In the Hugging Face implementation the position tensor is `[3, B, T]`, and for pure text the three axes hold the same positions.

## The code: three functions {#代码三个函数}

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

**`rope_tables`: the angle of every position and every pair.**

1. `rot = 64`: the number of dimensions that rotate.
2. `inv_freq`: the 32 speeds $\theta_0 \dots \theta_{31}$. `torch.arange(0, rot, 2) / rot` is 0/64, 2/64, …, 62/64, the $2i/d$ of the formula.
3. `positions[..., None] * inv_freq`: every position times every speed gives the angles, of shape `[T, 32]`.
4. `torch.cat([angles, angles])`: dimensions j and j + 32 use the same angle, so the 32 angles are repeated, giving `[T, 64]`.
5. It returns two tables, cos and sin. They depend only on the positions, not on the content of the input, so the whole model computes them once per forward pass and the 6 full-attention layers share them.

**`rotate_half`**: swaps the two halves of x and puts a minus sign on the front half, i.e. $(a, b) \to (-b, a)$.

**`apply_rope`**: computes `x * cos + rotate_half(x) * sin` on the first 64 dimensions only, which pair by pair is the 2D rotation; the last 192 dimensions are concatenated back unchanged. cos and sin broadcast over the head dimension, so all heads at the same position turn by the same angles.

**Who calls it.** In chapter 6's attention, q and k each call `apply_rope` once after their QK-Norm; v is not rotated, because v takes no part in scoring. The 18 linear-attention layers (chapter 7) do not use RoPE.

## Run it: from 2D to Qwen3.5's 32 pairs {#跑起来从两维到-qwen35-的-32-对}

```python title="ch05_rope.py" ci="loose"
"""第 5 章：先在两维里转一个角，再看 Qwen3.5 的 32 对各转多快、哪两维是一对、点积为什么只看距离"""
import math
import os

import torch

from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))


def rotate2d(v, angle):                                           # 1. turning by an angle in 2D: (a, b) -> (a cos - b sin, a sin + b cos)
    a, b = v
    return (a * math.cos(angle) - b * math.sin(angle), a * math.sin(angle) + b * math.cos(angle))


step = math.radians(30)                                           # a toy setting: 30 degrees more for every position further along
for pq, pk in ((3, 1), (103, 101), (3, 2)):
    rq, rk = rotate2d((1.0, 0.0), pq * step), rotate2d((1.0, 0.0), pk * step)
    print(f"q 在 {pq:3d}（转到 {pq * 30 % 360:3d} 度），k 在 {pk:3d}（转到 {pk * 30 % 360:3d} 度）：夹角 {(pq - pk) * 30:2d} 度，点积 {rq[0] * rk[0] + rq[1] * rk[1]:.3f}")

cfg = Config.load(find_model())
rot = int(cfg.head_dim * cfg.partial_rotary_factor)
inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))   # 2. Qwen3.5's 32 pairs: how fast each pair turns
print(f"每个头 {cfg.head_dim} 维，前 {rot} 维参与旋转，共 {rot // 2} 对；后一对比前一对慢 {inv_freq[0] / inv_freq[1]:.2f} 倍")
for i in (0, 1, 2, 8, 16, 24, 31):
    print(f"  第 {i:2d} 对：每个位置转 {inv_freq[i]:.2e} 弧度，转满一圈要 {2 * math.pi / inv_freq[i]:.3g} 个位置")
print(f"基频换成经典的 10000：最慢的一对 {2 * math.pi * 1e4 ** ((rot - 2) / rot):.3g} 个位置就转满一圈，比 262144 的上下文短")   # 262144 is the maximum length in config.json

x = torch.zeros(1, 1, 1, cfg.head_dim)                            # 3. which two dims form a pair: put a single 1 in dim 0 and turn it at position 1
x[..., 0] = 1.0
cos, sin = rope_tables(cfg, torch.tensor([1]))
y = apply_rope(x, cos, sin)[0, 0, 0]
print("转完之后不为 0 的维度：", {j: round(y[j].item(), 4) for j in range(cfg.head_dim) if y[j].abs() > 1e-6}, f"= cos 1 和 sin 1：第 0 维和第 {rot // 2} 维是一对")

torch.manual_seed(0)                                              # 4. relative position: for random q and k, the same distance gives the same dot product
q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)


def dot(a, b, pa, pb):
    ca, sa = rope_tables(cfg, torch.tensor([pa]))
    cb, sb = rope_tables(cfg, torch.tensor([pb]))
    return (apply_rope(a, ca, sa) * apply_rope(b, cb, sb)).sum().item()


print(f"q 在 3、k 在 1：{dot(q, k, 3, 1):.4f}；q 在 103、k 在 101：{dot(q, k, 103, 101):.4f}；q 在 3、k 在 2：{dot(q, k, 3, 2):.4f}")

same = torch.zeros(1, 1, 1, cfg.head_dim)                         # 5. the same content gives a smaller dot product the farther apart it is (only the 64 rotated dims)
same[..., :rot] = 1.0
print("q 和 k 一样时，点积随距离：", ", ".join(f"{d}: {dot(same, same, d, 0):.1f}" for d in (0, 1, 10, 100, 1000, 10000, 100000)))

x = torch.randn(1, 6, 8, cfg.head_dim)                            # 6. partial rotation: the last 192 dims stay untouched
cos, sin = rope_tables(cfg, torch.arange(6))
y = apply_rope(x, cos, sin)
print(f"cos 表 {tuple(cos.shape)}；旋转后前 {rot} 维变了（最大差 {(y[..., :rot] - x[..., :rot]).abs().max():.2f}），后 {cfg.head_dim - rot} 维没动（最大差 {(y[..., rot:] - x[..., rot:]).abs().max():.1e}）")
```

```text title="输出"
q 在   3（转到  90 度），k 在   1（转到  30 度）：夹角 60 度，点积 0.500
q 在 103（转到 210 度），k 在 101（转到 150 度）：夹角 60 度，点积 0.500
q 在   3（转到  90 度），k 在   2（转到  60 度）：夹角 30 度，点积 0.866
每个头 256 维，前 64 维参与旋转，共 32 对；后一对比前一对慢 1.65 倍
  第  0 对：每个位置转 1.00e+00 弧度，转满一圈要 6.28 个位置
  第  1 对：每个位置转 6.04e-01 弧度，转满一圈要 10.4 个位置
  第  2 对：每个位置转 3.65e-01 弧度，转满一圈要 17.2 个位置
  第  8 对：每个位置转 1.78e-02 弧度，转满一圈要 353 个位置
  第 16 对：每个位置转 3.16e-04 弧度，转满一圈要 1.99e+04 个位置
  第 24 对：每个位置转 5.62e-06 弧度，转满一圈要 1.12e+06 个位置
  第 31 对：每个位置转 1.65e-07 弧度，转满一圈要 3.8e+07 个位置
基频换成经典的 10000：最慢的一对 4.71e+04 个位置就转满一圈，比 262144 的上下文短
转完之后不为 0 的维度： {0: 0.5403, 32: 0.8415} = cos 1 和 sin 1：第 0 维和第 32 维是一对
q 在 3、k 在 1：-20.3131；q 在 103、k 在 101：-20.3131；q 在 3、k 在 2：-19.6979
q 和 k 一样时，点积随距离： 0: 64.0, 1: 62.5, 10: 50.8, 100: 42.3, 1000: 40.1, 10000: 21.5, 100000: 19.3
cos 表 (6, 64)；旋转后前 64 维变了（最大差 5.38），后 192 维没动（最大差 0.0e+00）
```

Read it section by section:

1. **The 2D toy**: just like the hand calculation above, (3, 1) and (103, 101) are both at 60° with a dot product of 0.5; at distance 1 the dot product becomes 0.866.
2. **The speeds of the 32 pairs**: pair 0 turns once every 6.28 positions, pair 31 once every 38 million, each pair 1.65 times slower than the one before. With the classic base of 10,000, the slowest pair would complete a turn every 47,000 positions, shorter than the 262K context.
3. **The pairing**: put a single 1 in dimension 0; after the turn, dimension 0 holds cos 1 and dimension 32 holds sin 1, and every other dimension is 0. Dimensions 0 and 32 are a pair: the two halves are paired.
4. **Relative position**: with random 256-dimensional q and k, (3, 1) and (103, 101) still give exactly the same dot product, while (3, 2) does not.
5. **Farther apart, smaller dot product**: when q and k have the same content, the dot product at distance 0 is 64 (32 pairs, 2 each). As the distance grows, the fast pairs fall out of step first and cancel each other out, only the slow pairs still line up, and the dot product keeps falling. So the same content is easier to notice when it is close; this is only a rough trend, and a particular q and k need not decrease monotonically.
6. **Partial rotation**: the first 64 dimensions changed, the last 192 did not move at all.

!!! interview "How to explain it"
    Explain RoPE in three steps. First what it solves: attention scores look only at content and need position information, and additive position encodings mix absolute positions into the dot product. Then the method: turn q and k by $p\theta$ according to their positions, so the angle in the dot product is just $(m-n)\theta$ and depends only on the relative distance. Finally the set of speeds: a single speed wraps around, so many pairs are used, from fast to slow, like a second hand through an hour hand; the base decides how slow the slowest hand is, and so how long a distance can be told apart, and Qwen3.5 pairs 1e7 with a 262K context. Mentioning that Qwen3.5 rotates only the first quarter of the dimensions and pairs the two halves, and that M-RoPE collapses to one axis for pure text, shows you have actually read this generation's implementation.

## Summary {#小结}

- [x] Attention scores look only at content; RoPE turns q and k at position p by $p\theta$, so only the distance $m - n$ is left in the dot product.
- [x] A single speed wraps around, so 32 pairs with 32 speeds are used, from a turn every 6 positions to a turn every 38 million; the larger the base, the slower the slowest pair.
- [x] The first 64 dimensions rotate, and dimension j pairs with dimension j + 32 (`rotate_half`); the last 192 do not rotate.
- [x] The same content gives a roughly smaller dot product the farther apart it is.

**File created in this chapter**: `qwen/rope.py`.
