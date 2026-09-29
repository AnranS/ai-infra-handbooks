---
title: 第一个 kernel：向量加法与启动配置
chapter: basics/first-kernel.md
difficulty: 简单
tags: [gpusim, 启动配置, 边界检查]
---
实现两部分：

1. kernel `add_kernel(t, a, b, c, n)`：`c[i] = a[i] + b[i]`，每个线程处理一个元素；
2. 主机函数 `vector_add(a, b, block=256)`：`a`、`b` 是长度相同的 numpy 数组。把它们拷到设备上（`gs.to_device`），分配输出（`gs.empty`），
   计算 grid 大小并启动 kernel，最后把结果拷回主机（`.copy_to_host()`）返回。

grid 大小要**刚好够用**：`ceil(n / block)` 个 block；`n` 不是 `block` 的倍数时，最后一个 block 里多出来的线程不能越界访问。
`n = 0` 时直接返回空数组，不启动 kernel（CUDA 里 grid 为 0 的启动是错误）。

<!-- 题解 -->
`grid = gs.cdiv(n, block)`，也就是 `(n + block - 1) // block`。kernel 里 `if i < n:` 做边界检查。

gpusim 会检查越界访问（真实 GPU 上越界可能不报错，悄悄写坏别的数据）；`gs.empty` 分配的内存是"脏"的，没写到的位置读出来是 NaN。
CUDA C++ 版本在 WSL2 + NVIDIA GPU 上运行，会报告实际带宽：向量加法读 2 个、写 1 个 float，是典型的带宽瓶颈 kernel，好的实现能跑到显存带宽的 80% 以上。
