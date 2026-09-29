---
title: Triton 分块矩阵乘
chapter: tools/triton.md
difficulty: 困难
tags: [Triton, GEMM, tl.dot]
---
用 Triton 写 $C = AB$（`A` 是 `(M, K)`、`B` 是 `(K, N)`，行主序 float32）。每个 program 计算 C 的一个 `BM × BN` 块，沿 K 方向每次取 `BK`：

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr): ...

def matmul(a, b, BM=32, BN=32, BK=32):   # 返回 c；用 tritonkit.empty_like 系列函数分配输出（见下）
```

- 二维 grid：`(cdiv(M, BM), cdiv(N, BN))`，`pid_m = tl.program_id(0)`、`pid_n = tl.program_id(1)`；
- 用 `offs_m[:, None]`、`offs_n[None, :]` 这样的广播构造二维指针块；
- 所有 load 都要带 mask（M、N、K 都可能不是块大小的倍数），越界处 `other=0.0`；
- 累加器用 `tl.zeros((BM, BN), dtype=tl.float32)`，每次 `acc += tl.dot(a_block, b_block)`；
- 输出分配：`c = tritonkit.empty((M, N), like=a)`。

测试检查结果正确（包括各种不整齐的形状）。在模拟器下还会检查：每个 program 对 A、B 各 load `cdiv(K, BK)` 次，并且确实用了 `tl.dot`。

<!-- 题解 -->
```python
pid_m, pid_n = tl.program_id(0), tl.program_id(1)
offs_m = pid_m * BM + tl.arange(0, BM)
offs_n = pid_n * BN + tl.arange(0, BN)
offs_k = tl.arange(0, BK)
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k0 in range(0, K, BK):
    a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K), other=0.0)
    b = tl.load(b_ptr + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
    acc += tl.dot(a, b)
tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
         mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

和 CUDA 的共享内存分块相比，Triton 把"块怎么放进共享内存、怎么分给线程、怎么用 Tensor Core"交给了编译器，你只描述块级的数据流。
正文还介绍了 L2 缓存友好的 program 排列（grouped ordering）和自动调优。
