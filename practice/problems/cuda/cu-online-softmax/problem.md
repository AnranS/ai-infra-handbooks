---
title: 行 softmax：online softmax 只读两遍
chapter: kernels/softmax-norm.md
difficulty: 中等
tags: [gpusim, softmax, online softmax, warp shuffle]
---
对 `(rows, cols)` 的矩阵逐行做 softmax。每行一个 block（`block = 128`，`grid = rows`），`cols` 可以远大于 128，每个线程负责 `threadIdx.x, threadIdx.x + 128, ...` 这些列。

朴素的三遍法：第一遍求最大值，第二遍求 $\sum e^{x - m}$，第三遍写出结果——输入要读三遍。
**online softmax** 把前两遍合成一遍：每个线程维护 `(m, s)`，遇到新元素 $x$ 时：

$$m' = \max(m, x),\qquad s' = s \cdot e^{m - m'} + e^{x - m'}$$

两个部分结果 $(m_1, s_1)$、$(m_2, s_2)$ 也能合并：$m = \max(m_1, m_2)$，$s = s_1 e^{m_1 - m} + s_2 e^{m_2 - m}$。

实现 kernel `softmax_rows(t, x, y, rows, cols)`：

1. 每个线程用 online 公式扫一遍自己负责的元素，得到 `(m, s)`；
2. block 内合并所有线程的 `(m, s)`（warp 内用 `shfl_down`，warp 之间用共享内存）；
3. 第二遍：`y = exp(x - m) / s`。

测试检查：结果正确（包括很大的输入值、`cols` 不是 128 的倍数、`cols < 128`）；**每个输入元素只从全局内存读两次**。

<!-- 题解 -->
合并函数：

```python
def combine(m1, s1, m2, s2):
    m = max(m1, m2)
    if m == -inf: return m, 0.0
    return m, s1 * exp(m1 - m) + s2 * exp(m2 - m)
```

warp 内：`m2 = yield t.shfl_down(m, off)`、`s2 = yield t.shfl_down(s, off)`，再 `combine`；每个 warp 的 lane 0 把结果写进共享内存 `sm[warp]`、`ss[warp]`，
屏障后第 0 个 warp 再合并一次，写回 `sm[0]`、`ss[0]`，再屏障，所有线程读出最终的 `m`、`s`。
没有元素的线程初始值是 `(-inf, 0)`，合并时要避免 `-inf - (-inf) = nan`。

FlashAttention 的核心正是这个合并公式：它让 softmax 可以分块计算。
