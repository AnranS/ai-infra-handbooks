---
title: 估算：分块配置与共享内存
chapter: tools/tilelang.md
difficulty: 简单
tags: [估算,共享内存,流水线,自动调优]
---
给 GEMM 选块大小时，共享内存是最硬的约束。实现：

1. `shmem_bytes(block_m, block_n, block_k, stages, dtype_bytes=2)`：一个 block 需要的共享内存字节数（A 块 `block_m × block_k` 和 B 块 `block_k × block_n`，各 `stages` 份）；
2. `blocks_per_sm(block_m, block_n, block_k, stages, smem_per_sm=228*1024, dtype_bytes=2)`：一个 SM 能驻留几个这样的 block；
3. `accum_regs(block_m, block_n, threads)`：累加器（FP32）平均分到每个线程要几个 32 位寄存器；
4. `is_feasible(block_m, block_n, block_k, stages, threads, smem_per_sm=228*1024, max_regs=255)`：共享内存至少放得下一个 block、且累加器寄存器不超过每线程上限时为 `True`。

```python
shmem_bytes(128, 128, 64, 3)                    # 98304
blocks_per_sm(128, 128, 64, 3)                  # 2
accum_regs(128, 128, 256)                       # 64
is_feasible(128, 128, 64, 3, 256)               # True
is_feasible(256, 256, 64, 4, 128)               # False
```

<!-- 题解 -->
`shmem_bytes = (block_m * block_k + block_k * block_n) * dtype_bytes * stages`。流水级数是线性的乘数：3 级流水的 128×128×64 FP16 块要 96 KB，H100 每 SM 228 KB 只能放两个。

累加器在寄存器里：`block_m * block_n` 个 FP32，平均分给 `threads` 个线程，每个线程 `block_m * block_n / threads` 个寄存器。128×128 配 256 线程是 64 个，还算宽裕；256×256 配 128 线程就是 512 个，远超每线程 255 的上限——这时就必须靠 Blackwell 的 Tensor Memory 或者把块切小。

自动调优搜的就是这几个参数的组合：块太小喂不饱 Tensor Core、块太大放不下；流水级数太少藏不住搬运延迟、太多挤占共享内存。这道题给出的是可行域，真正选哪一个要实测。
