---
title: 估算：分块 GEMM 的全局访存量与共享内存预算
chapter: kernels/gemm.md
difficulty: 中等
tags: [估算, GEMM, 共享内存, 算术强度]
---
分块 GEMM 里，每个 `BM × BN` 的输出 tile 要把 A 的 `BM` 行和 B 的 `BN` 列沿 K 方向完整读一遍。
于是整个 GEMM 里 **A 被读了 $\lceil N/BN \rceil$ 遍，B 被读了 $\lceil M/BM \rceil$ 遍**（假设没有 L2 复用）。tile 越大，全局访存越少，但共享内存和寄存器用得越多。

实现（输入元素 `in_bytes` 字节，输出元素 `out_bytes` 字节）：

1. `global_bytes(M, N, K, BM, BN, in_bytes=2, out_bytes=2)`：$(MK\lceil N/BN\rceil + KN\lceil M/BM\rceil) \cdot \text{in\_bytes} + MN \cdot \text{out\_bytes}$；
2. `tiled_intensity(M, N, K, BM, BN, in_bytes=2, out_bytes=2)`：$2MNK$ ÷ `global_bytes`；
3. `smem_bytes(BM, BN, BK, stages, in_bytes=2)`：多级流水线下每个块的共享内存 $(BM \cdot BK + BK \cdot BN) \cdot \text{in\_bytes} \cdot \text{stages}$；
4. `max_blocks_by_smem(smem_per_block, smem_per_sm=228 * 1024, reserved=1024)`：只看共享内存时每个 SM 最多驻留的块数（每个块额外占 `reserved` 字节系统保留）。

```python
global_bytes(4096, 4096, 4096, 128, 128)       # 约 2.18e9 字节
tiled_intensity(4096, 4096, 4096, 128, 128)    # 约 63 FLOPs/Byte
smem_bytes(128, 128, 64, 3)                     # 98304：96 KiB
max_blocks_by_smem(98304)                       # 2
```

<!-- 题解 -->
`4096³` 用 128 × 128 的 tile，A、B 各被读 32 遍，全局访存约 2.18 GB，算术强度约 63，远低于 H100 的屋脊点（约 295）。
可 cuBLAS 在这个形状上明明是算力瓶颈——差别在 **L2 复用**：同时在跑的块如果读的是相邻的 A 行、B 列，重复的读取会在 L2 命中，
真正落到 HBM 的流量要小得多。这就是 GEMM kernel 要做 tile 光栅化（swizzle）、让相邻块共享输入的原因。

tile 放大一倍（128 × 256），A 的读取次数减半，算术强度上升；代价是共享内存和寄存器翻倍、能同时驻留的块变少、波次量化更明显——tile 大小永远是折中。
