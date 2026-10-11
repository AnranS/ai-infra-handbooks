---
title: 只转一部分维度的 RoPE
chapter: rope.md
difficulty: 中等
tags: [RoPE, 位置编码, 相对位置, 广播]
requires: [torch]
---
Qwen3.5 的每个注意力头 256 维，只旋转前 64 维（`partial_rotary_factor = 0.25`），基频 $10^7$。写两个函数：

1. `rope_tables(positions, rot_dim, theta)`：`positions` 是 `[T]` 的整数位置，返回 `cos, sin`，形状各 `[T, rot_dim]`。频率 $\theta_i = \text{theta}^{-2i/\text{rot\_dim}}$（$i = 0 \ldots \text{rot\_dim}/2 - 1$），角度 = 位置 × 频率；前后两半配对，所以表的后一半重复前一半；
2. `apply_rope(x, cos, sin)`：`x` 形状 `[B, T, 头数, D]`，只旋转前 `rot_dim` 维（rotate-half 风格：前半 $a$、后半 $b$ 变成 $(a\cos - b\sin,\; b\cos + a\sin)$），剩下的 `D - rot_dim` 维原样拼回。

```python
cos, sin = rope_tables(torch.arange(6), 64, 1e7)
cos.shape                                 # torch.Size([6, 64])
x = torch.randn(1, 6, 8, 256)
apply_rope(x, cos, sin)[..., 64:].equal(x[..., 64:])   # True：后 192 维没动
```

<!-- 题解 -->
```python
def rope_tables(positions, rot_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    half = rot // 2
    rotated = torch.cat([-x_rot[..., half:], x_rot[..., :half]], dim=-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat([x_rot * cos + rotated * sin, x_pass], dim=-1)
```

测试会检查 RoPE 的"相对性"：同一对 q、k 放在位置 (3, 1) 和 (103, 101)，点积相同——因为旋转矩阵满足 $R_p^\top R_{p'} = R_{p' - p}$。没被旋转的那 192 维不带位置信息，点积与位置无关，这正是部分旋转的用意：留一些维度做"和位置无关的匹配"。

常见错误：把 `cos` 广播到 `[B, T, 头, D]` 时少了头那一维（`cos[None, :, None, :]` 四维才能对上）；或者 rotate-half 的符号写反——写反了"相对性"测试会过不了。
