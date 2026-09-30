---
title: 估算：屋脊点与代际差异
chapter: arch/evolution.md
difficulty: 简单
tags: [估算, 屋脊点, 量化, decode]
---
算力涨得比带宽快，屋脊点（算力 ÷ 带宽，单位 FLOP/字节）一代比一代高。实现四个函数（算力单位 TFLOPS、带宽单位 GB/s，1 TFLOPS = 1e12 FLOP/s，1 GB/s = 1e9 字节/s）：

1. `ridge(tflops, gbs)`：屋脊点；
2. `breakeven_batch(tflops, gbs, weight_bytes=2)`：decode 时权重被整个 batch 共用，读 1 字节权重做 `2 × batch` 次运算，batch 要多大才达到屋脊点（返回浮点数）；
3. `attn_intensity(group_size, kv_bytes=2)`：注意力部分的算术强度。每个 KV 元素被组里 `group_size` 个查询头各用一次乘加（2 FLOP），KV 每个元素占 `kv_bytes` 字节；
4. `decode_floor_ms(weight_gb, gbs)`：batch 为 1 时 decode 一步的下限（毫秒），即把权重读一遍的时间。

```python
ridge(989, 3350)                      # 295.2：H100 BF16
breakeven_batch(989, 3350)            # 295.2：BF16 权重要 batch 295 才算力受限
breakeven_batch(989, 3350, 0.5)       # 73.8：4 比特权重只要 74
attn_intensity(8)                     # 8.0：GQA 8 组，远低于屋脊点
decode_floor_ms(16, 3350)             # 4.78：16 GB 权重 / 3.35 TB/s
```

<!-- 题解 -->
`ridge = tflops * 1e12 / (gbs * 1e9)`。`breakeven_batch`：算术强度是 `2 × batch / weight_bytes`，令它等于屋脊点，得到 `batch = ridge × weight_bytes / 2`。BF16 权重时正好等于屋脊点，量化到 4 比特后门槛降到四分之一——**量化不但省显存，还降低了变成算力受限的门槛**。

`attn_intensity` 与 batch 无关：KV 是每个请求自己的，加大 batch 不会让某个请求的 KV 被别人复用，强度只等于 `2 × group_size / kv_bytes`，BF16 时就是 `group_size`。MHA 是 1、GQA 8 组是 8、MLA 吸收后约 128，都远低于屋脊点，所以 **decode 的注意力部分永远是带宽受限的**，只能靠减少 KV 字节（GQA、MLA、量化 KV、稀疏注意力）来加速。

`decode_floor_ms` 是 decode 速度的硬下限：H100 上 16 GB 权重要 4.8 ms，所以 batch 为 1 时每秒最多约 200 个 token。
