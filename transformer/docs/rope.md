# （五）RoPE：旋转位置编码

<p class="lead">注意力本身不知道 token 的先后——把一句话打乱，每对 (query, key) 的点积不变。RoPE 把"第几个位置"编成一个角度，<strong>把 query 和 key 转过这个角</strong>，于是两者的点积只和它们的距离有关。Qwen3.5 的版本有三个不常见的细节：只旋转每个头前 1/4 的维度、基频 10,000,000、以及为图像和视频准备的三轴位置（M-RoPE）。这一章把它写出来，看每一对转得多快、相对位置怎么出现在点积里。</p>

!!! note "这一章跑什么"
    `ch05_rope.py`，约 10 秒。

@@files@@

## 转一个角

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

把每个头的 256 维看成 128 对 $(a, b)$，对第 $i$ 对，位置 $p$ 对应的转角是 $p \cdot \theta_i$，其中 $\theta_i = \text{base}^{-2i/d}$ 从 1 一路降到 $1/\text{base}$——前几对转得快（分辨相邻位置），后几对转得极慢（分辨几千几万个位置外的远近）。两个被转过的向量的点积：

$$
\langle R_{p}\,q,\; R_{p'}\,k \rangle = \langle q,\; R_{p'-p}\,k \rangle
$$

只依赖 $p' - p$。这就是 RoPE 的全部：绝对位置加在向量上，相对位置出现在点积里。

**只旋转 1/4 维。** `partial_rotary_factor = 0.25`：256 维里只有前 64 维（32 对）被旋转，其余 192 维不带位置信息。这是 Qwen3.5 从 Qwen3-Next 继承的做法：留一部分维度专门做"和位置无关的匹配"，长上下文下更稳。

**基频 10,000,000。** 经典值是 10,000；基频越大，最慢的那几对转得越慢，262K 的上下文里才不会"转过一圈"把远近混淆。

**M-RoPE。** 多模态时位置有三个轴：时间（文本的先后、视频的帧）、图像的高、图像的宽。32 个频率按 `mrope_section = [11, 11, 10]` 交错分给三个轴（第 0、3、6 … 个频率归时间轴，第 1、4、7 … 归高，第 2、5 … 归宽）。纯文本时三个轴的位置相同，三张表完全一样，所以上面的代码直接按一个轴写（Hugging Face 的实现里位置张量是 `[3, B, T]`，纯文本时三个轴装的是同一份位置）。

## 跑起来：转得多快、相对位置、没转的维度

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

inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))   # 1. 每一对转多快：转满一圈要多少个位置
for i in (0, 8, 16, 24, 31):
    print(f"  第 {i:2d} 对：每个位置转 {inv_freq[i]:.2e} 弧度，转满一圈要 {2 * math.pi / inv_freq[i]:.3g} 个位置")

torch.manual_seed(0)                                              # 2. 相对位置：同一对 q、k，放在不同位置但距离相同，点积相同
q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)


def dot(pq, pk):
    cq, sq = rope_tables(cfg, torch.tensor([pq]))
    ck, sk = rope_tables(cfg, torch.tensor([pk]))
    return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()


print(f"q 在 3、k 在 1：{dot(3, 1):.4f}；q 在 103、k 在 101：{dot(103, 101):.4f}；q 在 3、k 在 2：{dot(3, 2):.4f}")
print(f"距离从 0 拉到 2000，点积怎么变：", [f"{dot(d, 0):.1f}" for d in (0, 1, 2, 10, 100, 2000)])

x = torch.randn(1, 6, 8, cfg.head_dim)                            # 3. 部分旋转：后 192 维原样不动
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

第 0 对 6.28 个位置就转满一圈，第 31 对要 3800 万个位置——快的对分辨相邻位置，慢的对分辨几千几万个位置外的远近，262K 的上下文在最慢的几对眼里还没转过四分之一圈。中间那行是 RoPE 的"相对性"：(3, 1) 和 (103, 101) 的点积一样，因为距离都是 2；(3, 2) 不一样，因为距离变了；距离从 0 拉到 2000，点积一路在变。最后一行：后 192 维原样不动。

!!! interview "怎么讲清楚"
    讲 RoPE 先说它解决什么：注意力对顺序不敏感，需要位置信息，而加法式的位置编码让点积里混进"绝对位置"项。再说做法：按位置转角，点积只剩相对位置。最后说两个数字的含义：基频决定能分辨多长的距离（Qwen3.5 用 1e7 配 262K 上下文），部分旋转留一些维度不带位置。能顺手提一句 M-RoPE——三轴位置、纯文本退化为一轴——说明你看过这一代多模态模型的实现。

## 小结

- [x] RoPE 把位置 $p$ 编成转角 $p\theta_i$，旋转 query 和 key；点积只依赖相对距离。
- [x] Qwen3.5：只转每头前 64 维、基频 1e7、M-RoPE 三轴（纯文本退化成一轴）；最快的一对 6 个位置转一圈，最慢的一对 3800 万个。

**这一章新建的文件**：`qwen/rope.py`。
