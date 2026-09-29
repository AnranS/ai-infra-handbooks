---
title: 并行归约：顺序寻址与加载时先加一次
chapter: kernels/reduction.md
difficulty: 中等
tags: [gpusim, 归约, bank conflict]
---
正文里归约的七个版本，这里实现其中两步关键优化，再把多个 block 的部分和串起来。

**kernel** `reduce_kernel(t, x, partial, n)`（`block = 256`）：每个 block 处理 **512** 个元素：

1. 线程 `tid` 读 `x[base + tid]` 和 `x[base + tid + 256]`（越界的算 0），**加起来**再写进共享内存 `s[tid]`（"加载时先加一次"，让 grid 减半）；
2. **顺序寻址**的树形归约：`stride` 从 128 开始每次减半，`tid < stride` 的线程做 `s[tid] += s[tid + stride]`，每轮之后屏障；
3. 线程 0 把 `s[0]` 写到 `partial[blockIdx.x]`。

**主机函数** `reduce_sum(x)`：`x` 是 float32 的 numpy 数组。第一轮用 `cdiv(n, 512)` 个 block 得到部分和，如果部分和不止一个，就对部分和数组再启动一轮，直到只剩一个数，返回它（Python `float`）。`n = 0` 时返回 `0.0`。

测试检查：结果正确；**没有 bank conflict**（交错寻址 `s[2*stride*tid] += ...` 会有）；第一轮启动的 block 数正好是 `cdiv(n, 512)`。

<!-- 题解 -->
交错寻址（`index = 2 * stride * tid`）让同一个 warp 的线程访问间隔 `2 * stride` 的地址，stride 为 16 时 32 个线程落在同一个 bank 的不同地址上，冲突严重；
顺序寻址（`s[tid] += s[tid + stride]`）时 warp 内的线程访问连续地址，没有冲突，同时活跃的线程也集中在前面的 warp 里，减少了分支发散。

多轮：`while len(cur) > 1: launch(...); cur = partial`，每轮数据量缩小 512 倍，一亿个元素也只要三轮。
CUDA C++ 版本在真卡上运行，会报告带宽：好的归约能跑到显存带宽的 80% 以上。
