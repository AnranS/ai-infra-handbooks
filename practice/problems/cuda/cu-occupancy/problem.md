---
title: 占用率计算器
chapter: basics/execution.md
difficulty: 中等
tags: [占用率, 资源限制, 估算]
---
一个 SM 能同时驻留多少个 block，取决于四种资源里最紧张的那一个。实现 `occupancy(threads_per_block, regs_per_thread, smem_per_block, gpu)`，
`gpu` 是字典：

```python
H100 = {"max_threads": 2048, "max_blocks": 32, "regs": 65536, "smem": 228 * 1024, "max_warps": 64,
        "reg_alloc_unit": 256, "smem_alloc_unit": 128, "smem_per_block_reserved": 1024}
```

计算规则（和 NVIDIA 占用率计算器一致的简化版）：

- **线程**：`max_threads // threads_per_block`；
- **block 数**：`max_blocks`；
- **寄存器**：寄存器按 warp 分配，每个 warp 用 `ceil(regs_per_thread * 32 / reg_alloc_unit) * reg_alloc_unit` 个；
  一个 block 的 warp 数是 `ceil(threads_per_block / 32)`，能放下的 block 数是 `regs // (每 warp 寄存器 * 每 block warp 数)`；`regs_per_thread = 0` 时不受限；
- **共享内存**：每个 block 实际占用 `ceil((smem_per_block + smem_per_block_reserved) / smem_alloc_unit) * smem_alloc_unit` 字节，能放下的 block 数是 `smem // 这个值`；
- 每 SM 的 block 数取四者最小值；占用率 = 驻留的 warp 数 / `max_warps`。

返回 `(blocks_per_sm, occupancy, limiter)`，`limiter` 是最紧张的资源名：`"threads"`、`"blocks"`、`"registers"`、`"shared_memory"` 之一（有多个并列最小时按这个顺序取第一个）。
`threads_per_block` 超过 1024 或者单个 block 就放不下（任何一项为 0）时，返回 `(0, 0.0, 对应的资源名)`。

<!-- 题解 -->
每项算一个上限，取最小值：

```python
warps = ceil(threads / 32)
regs_per_warp = ceil(regs * 32 / unit) * unit
limits = {"threads": max_threads // threads, "blocks": max_blocks,
          "registers": gpu_regs // (regs_per_warp * warps) if regs else inf,
          "shared_memory": gpu_smem // ceil_to(smem + reserved, smem_unit)}
```

正文练习里的例子（128 线程、168 个寄存器、64 KB 共享内存）：寄存器每 warp 5376 → 每 block 21504 → 3 个 block；
共享内存每 block 65 KB → 3 个 block；两者并列时按规定的顺序报告 `registers`。占用率 12 / 64 = 18.75%。
