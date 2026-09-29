---
title: 估算：GEMM 的波次量化与 tile 选择
chapter: kernels/gemm.md
difficulty: 中等
tags: [估算, GEMM, 波次量化, tile]
---
一个 GEMM 被切成若干个 `BM × BN` 的输出 tile，每个线程块算一个 tile。GPU 一次最多同时跑 `n_sms × blocks_per_sm` 个块，
所有 tile 要分成若干"波"（wave）才能跑完。最后一波如果只有少数几个块，大部分 SM 就空着——这叫**波次量化**（wave quantization），
是"形状稍微一变，性能掉一截"的常见原因。

实现：

1. `waves(M, N, BM, BN, n_sms, blocks_per_sm=1)`：返回 `(tiles, n_waves, efficiency)`：
   `tiles` $= \lceil M/BM \rceil \cdot \lceil N/BN \rceil$；`n_waves` $= \lceil \text{tiles} / (\text{n\_sms} \cdot \text{blocks\_per\_sm}) \rceil$；
   `efficiency` = `tiles` ÷（`n_waves` × `n_sms` × `blocks_per_sm`），即所有波次里 SM 的平均利用率；
2. `best_tile(M, N, candidates, n_sms, blocks_per_sm=1)`：从 `candidates`（`(BM, BN)` 的列表）里选 `efficiency` 最高的；
   相同时选 tile 面积更大的（数据复用更好）；仍相同时选列表里靠前的。

```python
waves(4096, 4096, 128, 128, 132)     # (1024, 8, 0.9697)：H100 有 132 个 SM
waves(1536, 1536, 128, 128, 132)     # (144, 2, 0.545)：第二波只有 12 个块，SM 利用率只有一半多
best_tile(1536, 1536, [(128, 128), (128, 256), (64, 128)], 132)   # (64, 128)
```

<!-- 题解 -->
`1536 × 1536` 切成 128 × 128 的 tile 恰好 144 个，比 132 个 SM 多 12 个，于是要跑两波，第二波只有 12 个 SM 在干活，整体利用率 54.5%。
换成 64 × 128 的 tile，288 个块正好接近 3 波（396 个位置），利用率 72.7%。

实际的 GEMM 库（cuBLAS、CUTLASS）会根据形状挑 tile，或用 split-K、stream-K 把最后一波的工作拆细，就是为了解决这个问题。
面试时如果被问到"为什么 M 从 1024 变到 1040，GEMM 变慢了"，波次量化是首先要想到的原因之一。
