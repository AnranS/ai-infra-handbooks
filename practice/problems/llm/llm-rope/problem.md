---
title: RoPE：两种写法与权重重排
chapter: transformer/position.md
difficulty: 中等
tags: [RoPE, 位置编码, 权重转换]
---
RoPE 把 query、key 的每两个维度看成一个二维向量，按位置旋转。第 $i$ 组的角频率 $\theta_i = \text{base}^{-2i/d_h}$，$i = 0, \dots, d_h/2 - 1$。

维度怎么分组有两种写法（见正文）：

- **前后两半一组**（Hugging Face / Qwen）：第 $i$ 维和第 $i + d_h/2$ 维一组；
- **相邻两维一组**（Meta 原始 LLaMA）：第 $2i$ 维和第 $2i+1$ 维一组。

一组 $(a, b)$ 在位置 $m$ 旋转后是 $(a\cos m\theta_i - b\sin m\theta_i,\; a\sin m\theta_i + b\cos m\theta_i)$。

实现（`x` 形状 `(T, H, d_h)`，`positions` 形状 `(T,)` 的整数数组，表示每个 token 在序列里的位置）：

1. `rope_half(x, positions, base=10000.0)`；
2. `rope_interleaved(x, positions, base=10000.0)`；
3. `half_perm(d_h)`：返回一个排列 `perm`（整数数组），满足对任意 `x`：
   `rope_half(x[..., perm], pos) == rope_interleaved(x, pos)[..., perm]`。
   把 Meta 格式的 checkpoint 转成 Hugging Face 格式时，就要按这个排列重排 `q_proj`、`k_proj` 的输出维度。

<!-- 题解 -->
```python
inv = base ** (-np.arange(0, dh, 2) / dh)          # θ_i
ang = positions[:, None] * inv[None, :]            # (T, dh/2)
cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]   # 广播到头维
a, b = x[..., :dh//2], x[..., dh//2:]              # 前后两半
out = concat([a*cos - b*sin, a*sin + b*cos])
```

相邻写法用 `x[..., 0::2]`、`x[..., 1::2]` 取出每组的两个分量，旋转后再交错放回。

排列：相邻写法的第 $i$ 组是 $(2i, 2i+1)$，要变成前后两半写法的 $(i, i + d_h/2)$，
所以 `perm = [0, 2, 4, ..., d_h-2, 1, 3, 5, ..., d_h-1]`。

RoPE 的关键性质：$\langle \mathrm{rope}(q, m), \mathrm{rope}(k, n) \rangle$ 只取决于 $m - n$，测试里会检查这一点。
