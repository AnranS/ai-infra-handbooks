---
title: 估算：Tensor Core 的分块与操作数带宽
chapter: arch/gpu-sm.md
difficulty: 中等
tags: [估算, Tensor Core, 共享内存, Tensor Memory]
---
Tensor Core 每代吞吐翻倍，而共享内存的带宽一直是每 SM 每周期 128 字节，所以一条矩阵乘指令算的块必须越来越大。一条指令算 `M × N` 的块、K 方向走一步，要读 `(M + N) × 2` 字节的 16 位操作数（A 的 M 行、B 的 N 列），做 `M × N` 次乘加。实现：

1. `operand_bytes_per_fma(m, n, a_from_smem=True, b_from_smem=True)`：每次乘加要从共享内存读多少字节（某个操作数不从共享内存读时不计入）；
2. `smem_pressure(m, n, fma_per_cycle, smem_bw=128, **kw)`：Tensor Core 满速时每周期要读多少字节，返回 `(字节数, 占带宽的比例)`；
3. `accum_bytes(m, n, dtype_bytes=4)`：一块累加器多大（字节）；
4. `fits_in_registers(m, n, threads, regs_per_thread=255, dtype_bytes=4)`：这块累加器放进 `threads` 个线程的寄存器里，每个线程要几个 32 位寄存器，返回 `(每线程寄存器数, 是否放得下)`。

```python
smem_pressure(64, 64, 2048)          # (128.0, 1.0)：Hopper 用 64×64 的块要占满共享内存带宽
smem_pressure(64, 256, 2048)         # (80.0, 0.625)
accum_bytes(128, 256)                # 131072：128 KB
fits_in_registers(128, 256, 128)     # (256, False)：超过每线程 255 个寄存器的上限
```

<!-- 题解 -->
每次乘加分摊到的字节数是 `2(M + N) / (MN)`：块越大越省。Hopper 的吞吐是每 SM 每周期 2048 次乘加，64×64 的块要 `2048 × 0.0625 = 128` 字节/周期，正好占满共享内存带宽，别的读写就没位置了——所以 `wgmma` 把 N 扩大到 256，降到 62.5%。Blackwell 再翻倍到 4096，单个 SM 用 128×256 也要 75%，两个 SM 合作算 256×256 时每个 SM 只放一半的 B，才回到 50%。

块大了，累加器也跟着大：128×256 的 FP32 累加器是 128 KB，摊到一个 128 线程的 warpgroup 上每个线程要 256 个寄存器，超过了 255 的上限，还没算索引、地址这些别的用途。这就是 Blackwell 给每个 SM 加 256 KB Tensor Memory 的原因：累加器常驻在那里，正好放得下两块 128×256，一块在算、另一块在收尾。
