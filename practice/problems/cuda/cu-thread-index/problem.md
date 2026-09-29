---
title: 线程层级：全局下标、warp 与 lane
chapter: basics/gpu-architecture.md
difficulty: 简单
tags: [gpusim, 线程层级, warp]
---
用 [gpusim](#/local) 写一个 kernel `whoami(t, out)`：启动配置是二维的 `grid = (gx, gy)`、`block = (bx, by)`。
每个线程在 `out` 里自己的位置上写下 4 个数：

- `out[gid, 0]`：线程的**全局线性编号** `gid`：先按 block 编号（`blockIdx.y * gridDim.x + blockIdx.x`），block 内再按线程的线性编号（`threadIdx.y * blockDim.x + threadIdx.x`）；
- `out[gid, 1]`：所在 block 的线性编号；
- `out[gid, 2]`：在 block 内属于第几个 warp（每 32 个连续的线性编号是一个 warp）；
- `out[gid, 3]`：在 warp 内的 lane 编号（0～31）。

`out` 是形状 `(总线程数, 4)` 的 int32 数组。

gpusim 的内置变量：`t.threadIdx.x / .y`、`t.blockIdx.x / .y`、`t.blockDim.x / .y`、`t.gridDim.x / .y`，和 CUDA 一样。

<!-- 题解 -->
```python
tid = t.threadIdx.y * t.blockDim.x + t.threadIdx.x          # block 内线性编号
bid = t.blockIdx.y * t.gridDim.x + t.blockIdx.x             # block 线性编号
gid = bid * (t.blockDim.x * t.blockDim.y) + tid
```

warp 是按**线性编号**划分的：`block = (16, 8)` 时，第 0 个 warp 包含 `threadIdx.y` 为 0 和 1 的两行。
理解这一点很重要：二维 block 的 x 维太小（比如 `(4, 64)`）时，一个 warp 会跨很多行，访问二维数组时就可能不合并。
