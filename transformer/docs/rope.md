# （五）RoPE：旋转位置编码

<p class="lead">注意力只看内容，不看位置：把一句话的词序打乱，每对 (query, key) 的点积一个都不变。RoPE 的办法是<strong>按位置把 query 和 key 转一个角</strong>：排在第 p 个位置的向量转 p 份，于是两个向量之间的夹角里只剩下它们的距离。这一章先在两维里用具体数字转一遍，再看 Qwen3.5 怎么把它铺到 64 维、32 种转速上，最后逐行读代码。</p>

!!! abstract "学完这章你能"
    1. 用两维向量讲清 RoPE：按位置把 q、k 转一个角，点积里只剩两个位置的距离。
    2. 说出为什么要 32 对、32 种转速，以及 base（`rope_theta`）管的是什么。
    3. 指出一个头 256 维里哪两维是一对、哪些维不转，读懂 `rope_tables`、`rotate_half`、`apply_rope` 三个函数。

!!! note "这一章跑什么"
    `ch05_rope.py`，约 10 秒。

@@files@@

## 为什么注意力需要位置

第 6 章的注意力里，第 i 个 token 给第 j 个 token 打的分是 $q_i \cdot k_j$。q 和 k 都只由那个 token 自己的向量算出来，和它排在第几位没有关系。所以"狗咬人"和"人咬狗"里，"人"给"狗"打的分一模一样，模型分不出是谁咬谁。因果掩码只决定每个词能看见谁；看见的那些词谁先谁后，分数里体现不出来。

把位置告诉模型有两类办法：

- **加一个位置向量**：给第 p 个位置准备一个向量，加到词向量上（最早的 Transformer、GPT-2）。位置和内容混在一起，点积里会多出和"绝对位置"有关的项，而模型真正关心的往往是"隔了几个词"。
- **把 q 和 k 转一个角**：词向量不动，只在打分之前，按位置把 q 和 k 各转一个角。转完之后，分数只和两个位置的**差**有关。这就是 RoPE，现在的大模型几乎都用它。

## 先看两维：转一个角

一个二维向量 $(a, b)$ 逆时针转 $\varphi$ 角，变成

$$
(a, b) \;\to\; (a\cos\varphi - b\sin\varphi,\;\; a\sin\varphi + b\cos\varphi)
$$

长度不变，只有方向变了。RoPE 的规则只有一条：**排在第 p 个位置的向量，转 $p\theta$**，$\theta$ 是每往后一个位置多转的角度。

为什么这样一来点积就只看距离？点积等于两个长度相乘，再乘上夹角的余弦。q 在位置 m 转了 $m\theta$，k 在位置 n 转了 $n\theta$，它们之间的夹角就多了 $(m - n)\theta$，只和差 $m - n$ 有关。用具体数字算一遍，设 $\theta = 30°$，q 和 k 都是 $(1, 0)$：

- q 在位置 3、k 在位置 1：q 转到 90°，k 转到 30°，夹角 60°，点积 $\cos 60° = 0.5$。
- 两个都往后挪 100：q 在 103，转了 3090°，去掉整圈是 210°；k 在 101，转到 150°。夹角还是 60°，点积还是 0.5。
- q 在 3、k 在 2：距离变成 1，夹角 30°，点积 $\cos 30° = 0.866$。

![图：二维里转一个角：每往后一个位置多转 30°；q、k 一起往后挪 100，夹角和点积都不变](assets/figures/rope-2d.svg){.aig-svg}

写成公式，$R_\varphi$ 表示"转 $\varphi$ 角"：

$$
\langle R_{m\theta}\, q,\;\; R_{n\theta}\, k \rangle \;=\; \langle q,\;\; R_{(n-m)\theta}\, k \rangle
$$

右边只剩 $n - m$。绝对位置加在向量上，相对位置出现在点积里，这就是 RoPE 的全部想法。

## 一根指针不够：32 对、32 种转速

只用一个 $\theta$ 有个毛病：转满一圈就回到原处。$\theta = 30°$ 时，距离 12 的两个词夹角是 360°，和距离 0 完全一样，模型分不清"就是我自己"和"往前数第 12 个"。

解决办法和钟表一样。秒针一分钟转一圈，光看秒针，分不清现在和一分钟以前；再加上分针和时针，几根指针合起来就能确定时间。RoPE 把一个头的向量切成很多对，每一对是一根"指针"，各有各的转速：

$$
\theta_i = \text{base}^{-2i/d}, \qquad i = 0, 1, \dots, d/2 - 1
$$

Qwen3.5 每个头参与旋转的是 $d = 64$ 维，也就是 32 对，base 是 10,000,000：

- 第 0 对每个位置转 1 弧度，大约 57°，6.28 个位置转满一圈，像秒针，分辨"隔一两个词"。
- 往后每一对比前一对慢 1.65 倍。
- 第 31 对要 3800 万个位置才转一圈，像时针，分辨"隔几千几万个词"。

**base 为什么这么大。** 最慢的那根指针，在整个上下文里不能转满一圈，否则远处的距离又会和近处混在一起。Qwen3.5 的上下文是 262,144 个 token；如果用经典的 base = 10,000，最慢的一对 4.7 万个位置就转满一圈；换成 10,000,000，要 3800 万个位置。

下面的表盘可以拖。每个表盘是一对维度，蓝针是 q，橙针是 k。把 m、n 一起往后挪，每个表盘上的夹角都不变；只改其中一个，快的表盘转得飞快，慢的表盘几乎不动。

<div class="aig-widget" data-widget="rope-clock"></div>

## 256 维里哪两维是一对

Qwen3.5 每个头有 256 维，只有前 64 维参与旋转（`partial_rotary_factor = 0.25`），剩下 192 维原样不动，专门做和位置无关的匹配。这是 Qwen3.5 从 Qwen3-Next 继承的做法，长上下文下更稳。

前 64 维里怎么配对？不是相邻的 (0, 1)、(2, 3)……，而是**前一半配后一半**：第 j 维和第 j + 32 维是一对，用第 j 种转速。RoPE 的论文按相邻两维配对；Qwen 系列和 Hugging Face 里的大多数模型按前后两半配对。两种写法只差一个固定的维度排列，数学上等价；但权重是按哪种训练的，就得按哪种转，配错了模型的输出就乱了。

![图：一个头的 256 维：前 64 维参与旋转，第 j 维和第 j + 32 维是一对；后 192 维不转](assets/figures/rope-pairs.svg){.aig-svg}

前后两半配对时，"转一个角"可以写成两次逐元素乘法：先把 x 的前后两半对调、前面添个负号，得到 `rotate_half(x)`；再算 `x * cos + rotate_half(x) * sin`。设 $a$ 是第 j 维、$b$ 是第 j + 32 维，第 j 维得到 $a\cos - b\sin$，第 j + 32 维得到 $b\cos + a\sin$，正好是上面那个二维旋转。

**M-RoPE。** 多模态时位置有三个轴：时间（文本的先后、视频的帧）、图像的高、图像的宽。32 个频率按 `mrope_section = [11, 11, 10]` 交错分给三个轴：第 0、3、6 … 个频率归时间轴，第 1、4、7 … 归高，第 2、5 … 归宽。纯文本时三个轴的位置相同，三张表完全一样，所以下面的代码直接按一个轴写。Hugging Face 的实现里位置张量是 `[3, B, T]`，纯文本时三个轴装的是同一份位置。

## 代码：三个函数

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
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))   # 32 个频率：从 1 一路降到 1/theta
    angles = positions.float()[..., None] * inv_freq                                  # [T, 32]：位置 × 频率 = 转角
    angles = torch.cat([angles, angles], dim=-1)                                      # [T, 64]：前后两半配成一对，用同一个角
    return angles.cos(), angles.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    """x: [B, T, 头数, head_dim]。只转前 rot_dim 维，剩下的维度原样拼回去。"""
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    cos, sin = cos[..., None, :].to(x.dtype), sin[..., None, :].to(x.dtype)         # 对批和头广播；表是 fp32 算的，用的时候转成 x 的精度
    x_rot = x_rot * cos + rotate_half(x_rot) * sin                                    # 二维旋转：(a, b) -> (a cos - b sin, b cos + a sin)
    return torch.cat([x_rot, x_pass], dim=-1)
```

**`rope_tables`：算出每个位置、每一对的转角。**

1. `rot = 64`：参与旋转的维数。
2. `inv_freq`：32 种转速 $\theta_0 \dots \theta_{31}$。`torch.arange(0, rot, 2) / rot` 是 0/64、2/64……62/64，就是公式里的 $2i/d$。
3. `positions[..., None] * inv_freq`：每个位置乘每种转速，得到转角，形状 `[T, 32]`。
4. `torch.cat([angles, angles])`：第 j 维和第 j + 32 维用同一个角，所以把 32 个角重复一遍，变成 `[T, 64]`。
5. 返回 cos 和 sin 两张表。它们只和位置有关、和输入的内容无关，所以整模型每次前向只算一次，6 个全注意力层共用。

**`rotate_half`**：把 x 的前后两半对调，前面那半添个负号，也就是 $(a, b) \to (-b, a)$。

**`apply_rope`**：只取前 64 维做 `x * cos + rotate_half(x) * sin`，逐对展开就是二维旋转；后 192 维原样拼回去。cos、sin 在头这一维上广播，同一个位置上所有头转一样的角。

**谁来调用。** 第 6 章的注意力里，q 和 k 各自做完 QK-Norm 之后调一次 `apply_rope`；v 不转，因为 v 不参与打分。18 层线性注意力（第 7 章）不用 RoPE。

## 跑起来：从两维到 Qwen3.5 的 32 对

```python title="ch05_rope.py" ci="loose"
"""第 5 章：先在两维里转一个角，再看 Qwen3.5 的 32 对各转多快、哪两维是一对、点积为什么只看距离"""
import math
import os

import torch

from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))


def rotate2d(v, angle):                                           # 1. 两维里转一个角：(a, b) -> (a cos - b sin, a sin + b cos)
    a, b = v
    return (a * math.cos(angle) - b * math.sin(angle), a * math.sin(angle) + b * math.cos(angle))


step = math.radians(30)                                           # 玩具设定：每往后一个位置，多转 30 度
for pq, pk in ((3, 1), (103, 101), (3, 2)):
    rq, rk = rotate2d((1.0, 0.0), pq * step), rotate2d((1.0, 0.0), pk * step)
    print(f"q 在 {pq:3d}（转到 {pq * 30 % 360:3d} 度），k 在 {pk:3d}（转到 {pk * 30 % 360:3d} 度）：夹角 {(pq - pk) * 30:2d} 度，点积 {rq[0] * rk[0] + rq[1] * rk[1]:.3f}")

cfg = Config.load(find_model())
rot = int(cfg.head_dim * cfg.partial_rotary_factor)
inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))   # 2. Qwen3.5 的 32 对：每一对转多快
print(f"每个头 {cfg.head_dim} 维，前 {rot} 维参与旋转，共 {rot // 2} 对；后一对比前一对慢 {inv_freq[0] / inv_freq[1]:.2f} 倍")
for i in (0, 1, 2, 8, 16, 24, 31):
    print(f"  第 {i:2d} 对：每个位置转 {inv_freq[i]:.2e} 弧度，转满一圈要 {2 * math.pi / inv_freq[i]:.3g} 个位置")
print(f"基频换成经典的 10000：最慢的一对 {2 * math.pi * 1e4 ** ((rot - 2) / rot):.3g} 个位置就转满一圈，比 262144 的上下文短")   # 262144 是 config.json 里的最大长度

x = torch.zeros(1, 1, 1, cfg.head_dim)                            # 3. 哪两维是一对：只在第 0 维放一个 1，放到位置 1 上转一下
x[..., 0] = 1.0
cos, sin = rope_tables(cfg, torch.tensor([1]))
y = apply_rope(x, cos, sin)[0, 0, 0]
print("转完之后不为 0 的维度：", {j: round(y[j].item(), 4) for j in range(cfg.head_dim) if y[j].abs() > 1e-6}, f"= cos 1 和 sin 1：第 0 维和第 {rot // 2} 维是一对")

torch.manual_seed(0)                                              # 4. 相对位置：随机的 q、k，距离一样点积就一样
q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)


def dot(a, b, pa, pb):
    ca, sa = rope_tables(cfg, torch.tensor([pa]))
    cb, sb = rope_tables(cfg, torch.tensor([pb]))
    return (apply_rope(a, ca, sa) * apply_rope(b, cb, sb)).sum().item()


print(f"q 在 3、k 在 1：{dot(q, k, 3, 1):.4f}；q 在 103、k 在 101：{dot(q, k, 103, 101):.4f}；q 在 3、k 在 2：{dot(q, k, 3, 2):.4f}")

same = torch.zeros(1, 1, 1, cfg.head_dim)                         # 5. 同样的内容，离得越远点积越小（只看参与旋转的 64 维）
same[..., :rot] = 1.0
print("q 和 k 一样时，点积随距离：", ", ".join(f"{d}: {dot(same, same, d, 0):.1f}" for d in (0, 1, 10, 100, 1000, 10000, 100000)))

x = torch.randn(1, 6, 8, cfg.head_dim)                            # 6. 部分旋转：后 192 维原样不动
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

逐段对着看：

1. **两维的玩具**：和上面手算的一样，(3, 1) 和 (103, 101) 的夹角都是 60°，点积都是 0.5；距离变成 1，点积变成 0.866。
2. **32 对的转速**：第 0 对 6.28 个位置一圈，第 31 对 3800 万个位置一圈，每一对比前一对慢 1.65 倍。换成经典的 base = 10,000，最慢的一对 4.7 万个位置就转满一圈，比 262K 的上下文短。
3. **配对**：只在第 0 维放一个 1，转完之后第 0 维是 cos 1、第 32 维是 sin 1，别的维度都是 0。第 0 维和第 32 维就是一对，正是前后两半配对。
4. **相对位置**：换成随机的 256 维 q、k，(3, 1) 和 (103, 101) 的点积还是完全一样，(3, 2) 就不一样了。
5. **距离越远，点积越小**：q 和 k 内容一样时，距离 0 的点积是 64（32 对，每对贡献 2）。距离一拉开，快的那几对先转乱、互相抵消，只剩慢的几对还对得上，点积就一路往下掉。所以同样的内容，离得越近越容易被注意到；这只是个大致的趋势，具体的 q、k 不一定单调。
6. **部分旋转**：前 64 维变了，后 192 维一点没动。

## 常见疑问

**公式和代码怎么一一对上？**

| 公式 | 代码 |
| --- | --- |
| 第 i 对的转速 $\theta_i = \text{base}^{-2i/d}$（$d = 64$） | `inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))` |
| 位置 p 的转角 $p\,\theta_i$ | `angles = positions.float()[..., None] * inv_freq` |
| 第 j 维和第 j + 32 维用同一个角 | `torch.cat([angles, angles], dim=-1)` |
| 二维旋转 $(a, b) \to (a\cos\varphi - b\sin\varphi,\ b\cos\varphi + a\sin\varphi)$ | `x_rot * cos + rotate_half(x_rot) * sin` |
| 后 192 维不转 | `torch.cat([x_rot, x_pass], dim=-1)` |

**为什么 v 不转？**

位置只需要进到"打分"里：分数由 q 和 k 的点积决定，转 q、k 就够了。v 是被取走的内容，转了反而把位置混进内容里。

**论文里是相邻两维配对，这里为什么是前后两半配对？**

两种都对：只要 q 和 k 用同一种配对，点积只剩距离的性质就成立。但权重是按某一种配对训练出来的，Qwen 系列（和 HF 的实现）用的是前后两半配对，加载它的权重就必须用 `rotate_half`，换成相邻配对会全错。

!!! interview "怎么讲清楚"
    讲 RoPE 分三步。先说它解决什么：注意力的分数只看内容，需要位置信息；加法式的位置编码会让点积里混进绝对位置。再说做法：按位置把 q 和 k 转 $p\theta$，点积里的夹角只剩 $(m-n)\theta$，所以只和相对距离有关。最后说那组转速：一种转速会绕回来，所以用很多对、从快到慢，像秒针到时针；base 决定最慢的指针有多慢，也就决定了能分辨多长的距离，Qwen3.5 用 1e7 配 262K 上下文。能顺手说出 Qwen3.5 只转前 1/4 维、按前后两半配对，以及 M-RoPE 在纯文本时退化成一轴，说明你真读过这一代的实现。

## 小结

- [x] 注意力的分数只看内容；RoPE 把第 p 个位置的 q、k 转 $p\theta$，点积里只剩距离 $m - n$。
- [x] 一种转速会绕回来，所以用 32 对、32 种转速，从 6 个位置一圈到 3800 万个位置一圈；base 越大，最慢的那对越慢。
- [x] 前 64 维参与旋转，第 j 维和第 j + 32 维是一对（`rotate_half`）；后 192 维不转。
- [x] 同样的内容，距离越远点积大致越小。

## 思考题

**1. 把 base 从 1e7 换成 1e4，最慢的那一对转一圈要多少个位置？对 26 万的上下文有什么影响？**

??? success "参考答案"
    约 4.7 万个位置一圈（正文的实验算过）。26 万的上下文里，最慢的那对也绕了五圈多，相隔很远的两个位置可能转出几乎一样的角度，模型就分不清远近。所以要上长上下文，base 要调大（Qwen3.5 用 1e7）。

**2. q 在位置 m、k 在位置 n，两个位置都往后挪 1000，分数变不变？这对 KV cache 意味着什么？**

??? success "参考答案"
    不变，分数只和 $m - n$ 有关。k 的转角只取决于它自己的位置，算一次就定了，所以 KV cache 里存的是已经转好的 k；后面的 token 来了，只要转自己的 q，不用回头重转缓存里的 k。

**这一章新建的文件**：`qwen/rope.py`。
