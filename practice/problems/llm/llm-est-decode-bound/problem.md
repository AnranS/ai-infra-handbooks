---
title: 估算：decode 与 prefill 的时间下限
chapter: inference/estimation.md
difficulty: 中等
tags: [估算, decode, prefill, 张量并行]
---
上线前先算"物理极限"：decode 一步至少多久、每秒最多多少 token、首 token 至少等多久。约定：

- **decode 一步**：每张卡要读自己那份权重和本 batch 所有序列的 KV：
  字节 $= (\text{weight\_bytes} + \text{batch} \cdot \text{avg\_ctx} \cdot \text{kv\_bytes\_per\_token}) / \text{tp}$，时间 = 字节 ÷ 显存带宽；
  张量并行时每步还有 `n_allreduce` 次 all-reduce，每次按 `allreduce_us` 微秒算（decode 时数据量很小，主要是延迟）；
- **prefill**：算力瓶颈。FLOPs $= 2 \cdot N \cdot s + 2 \cdot L \cdot d_{attn} \cdot s^2$（第二项是因果注意力），
  时间 = FLOPs ÷（`tp` × 峰值 × `mfu`）。

实现（带宽单位 GB/s = $10^9$ 字节/秒，算力单位 TFLOPS，时间单位毫秒）：

1. `decode_step_ms(weight_bytes, kv_bytes_per_token, batch, avg_ctx, bw_gbs, tp=1, n_allreduce=0, allreduce_us=0.0)`；
2. `decode_tokens_per_s(weight_bytes, kv_bytes_per_token, batch, avg_ctx, bw_gbs, tp=1, n_allreduce=0, allreduce_us=0.0)`：整个实例每秒生成的 token 数上限；
3. `prefill_ms(n_params, prompt_len, n_layers, d_attn, peak_tflops, mfu=0.5, tp=1)`。

```python
# Llama-3-70B，BF16 权重 141 GB，8 × H100 张量并行，batch 32、平均上下文 4096
decode_step_ms(141e9, 327680, 32, 4096, 3350, tp=8)                              # 约 6.9 ms
decode_step_ms(141e9, 327680, 32, 4096, 3350, tp=8, n_allreduce=160, allreduce_us=10)   # 再加 1.6 ms
```

<!-- 题解 -->
- batch 变大时，读权重的时间被更多请求分摊，吞吐几乎线性上升；但 KV 读取量随 batch × 上下文线性增长，长上下文时它会超过权重，成为新的瓶颈；
- 张量并行把权重和 KV 都切开，单步时间下降，但每层两次 all-reduce 的延迟是固定开销：80 层就是 160 次，每次 10 微秒就是 1.6 ms，占到 decode 一步的 20%；
- prefill 是算力瓶颈，8K prompt 的注意力项已经不可忽略；把实际测到的时间和这个下限对比，就知道还有多少优化空间。
