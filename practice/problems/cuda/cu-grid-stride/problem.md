---
title: grid-stride 循环：用固定的 grid 处理任意长度
chapter: basics/execution.md
difficulty: 简单
tags: [gpusim, grid-stride, SAXPY]
---
实现 SAXPY：`y[i] = a * x[i] + y[i]`。这次 grid 的大小是**固定的**，和数据长度无关（真实场景里常按 SM 数量的若干倍启动，让每个线程处理多个元素）。

kernel `saxpy(t, a, x, y, n)` 用 **grid-stride 循环**：线程从自己的全局下标开始，每次前进"整个 grid 的线程总数"，直到超过 `n`。

测试会用各种 `grid × block` 组合（比数据多、比数据少、正好相等）启动它，结果都要正确；并检查访存是合并的（每次循环中，同一个 warp 读写的是连续的地址）。

<!-- 题解 -->
```python
i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
stride = t.gridDim.x * t.blockDim.x
while i < n:
    y[i] = a * x[i] + y[i]
    i += stride
```

另一种"每个线程处理连续的一段"（`i` 从 `gid * chunk` 到 `(gid+1) * chunk`）结果也对，但同一个 warp 的 32 个线程在同一时刻访问的地址相隔 `chunk` 个元素，不能合并，带宽会差很多。
