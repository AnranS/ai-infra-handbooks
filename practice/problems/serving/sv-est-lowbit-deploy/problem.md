---
title: 估算：低比特部署的显存与延迟下限
chapter: frontier/low-bit.md
difficulty: 简单
tags: [估算, 量化, INT4, FP4, MoE]
---
一个 MoE 模型总参数 `total`、每个 token 激活 `active` 个参数。实现：

1. `bytes_per_param(bits, group, scale_bytes)`：每个参数的平均字节数：`bits / 8`，加上每 `group` 个参数一个 `scale_bytes` 字节的缩放因子（`group=None` 表示没有缩放开销）；
2. `min_gpus(total, bpp, gpu_mem, reserve=0.3)`：权重放进 GPU（每张卡只能用 `(1 - reserve)` 的显存）至少要几张卡；
3. `decode_floor(active, bpp, gpus, bw)`：batch 为 1 时，`gpus` 张卡并行读激活权重，每个 token 的时间下限（秒）：`active × bpp / (gpus × bw)`；
4. `speedup_needed(bpp_old, bpp_new)`：从旧格式换成新格式后，decode 下限变成原来的几分之一（返回倍数 `bpp_old / bpp_new`）。

```python
bytes_per_param(4, 32, 2)                    # 0.5625（INT4，每 32 个一个 bf16 缩放）
min_gpus(1e12, 1.0, 141e9)                   # 11（FP8 的万亿模型要 11 张 H200）
```

<!-- 题解 -->
4 比特加上缩放因子的开销，通常在 0.53～0.57 字节 / 参数之间。万亿参数时，FP8 要两台 8 卡机，4 比特一台就够，跨机的专家并行变成机内——这比"省一半显存"影响大得多。decode 的延迟下限和读取的字节数成正比，4 比特相对 FP8 约快 1.8 倍（而不是 2 倍，因为缩放因子也要读）。
