---
title: 共享内存分块 GEMM
chapter: kernels/gemm.md
difficulty: 困难
tags: [gpusim, GEMM, 分块, 数据复用]
---
实现 $C = A B$（`A` 是 `(M, K)`、`B` 是 `(K, N)`，都是行主序的 float32）。朴素写法里每个线程读 A 的一行和 B 的一列，同一个元素被重复读了几十次。

用**共享内存分块**：`block = (T, T)`（`T = TILE = 8`，为了让模拟器跑得快，实际 GPU 上常用 16 或 32），`grid = (cdiv(N, T), cdiv(M, T))`，每个 block 计算 C 的一个 `T × T` 块：

```text
for k0 in range(0, K, T):
    协作地把 A[行块, k0:k0+T] 和 B[k0:k0+T, 列块] 读进共享内存 As、Bs（越界的填 0）
    __syncthreads()
    acc += As[ty, :] · Bs[:, tx]
    __syncthreads()        # 下一轮覆盖 As、Bs 之前，要等所有线程用完
```

实现 kernel `gemm_tiled(t, A, B, C, M, N, K)`。测试检查：结果正确（包括 M、N、K 不是 8 的倍数）；全局内存读的扇区数不超过朴素写法的 1/3；没有数据竞争（少了第二个屏障，gpusim 会报出来）。

<!-- 题解 -->
```python
As = t.shared("As", (T, T)); Bs = t.shared("Bs", (T, T))
row = t.blockIdx.y * T + ty; col = t.blockIdx.x * T + tx
acc = 0.0
for k0 in range(0, K, T):
    As[ty, tx] = A[row, k0 + tx] if row < M and k0 + tx < K else 0.0
    Bs[ty, tx] = B[k0 + ty, col] if k0 + ty < K and col < N else 0.0
    yield t.syncthreads()
    for k in range(T):
        acc += As[ty, k] * Bs[k, tx]
    yield t.syncthreads()
if row < M and col < N:
    C[row, col] = acc
```

每个元素从全局内存读进来后被 `T` 个线程复用，全局访存量降为原来的 $1/T$。
正文的后续版本（寄存器分块、向量化、双缓冲）在此基础上继续提高数据复用和指令级并行。
