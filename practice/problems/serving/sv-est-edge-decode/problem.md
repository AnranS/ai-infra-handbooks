---
title: 估算：端侧 decode 的速度上限与量化格式的选择
chapter: ops/edge.md
difficulty: 简单
tags: [估算, 端侧, GGUF, 量化]
---
单用户的 decode 每一步要把全部权重和当前上下文的 KV 读一遍，速度上限是"有效带宽 ÷ 每步读的字节数"。实现：

1. `bits_per_weight(fmt)`：GGUF 格式每个权重平均占多少比特（含缩放等元数据）：`F16` 16、`Q8_0` 8.5、`Q6_K` 6.5625、`Q4_K` 4.5、`Q4_1` 5.0、`Q4_0` 4.5；不认识的格式抛出 `ValueError`；
2. `kv_bytes(layers, kv_heads, head_dim, context, kv_bits=16)`：一个请求的 KV 字节数；
3. `decode_tps(weight_bytes, kv_bytes, bandwidth_gbs, efficiency=0.7)`：每秒最多生成多少个 token（带宽单位 GB/s，1 GB = 10⁹ 字节，实际能用到 `efficiency` 那么多）；
4. `pick_format(params, target_tps, bandwidth_gbs, efficiency=0.7)`：不考虑 KV，按质量从高到低 `F16 > Q8_0 > Q6_K > Q4_K > Q4_1 > Q4_0` 依次尝试，返回第一个能达到 `target_tps` 的格式，都达不到时返回 `None`；
5. `prefill_seconds(params, prompt_tokens, tflops, utilization=0.4)`：prefill 受算力限制，每个 token 约 `2 × params` 次运算，算力按峰值的 `utilization` 算。

```python
decode_tps(8e9 * bits_per_weight("Q4_K") / 8, 0, 60)     # 9.33：8B 模型 4 比特量化，手机 60 GB/s
pick_format(3e9, 20, 60)                                 # 'Q4_K'
```

<!-- 题解 -->
所有数字都来自同一个公式：每步读的字节数 = 权重字节数 + KV 字节数。手机的 60 GB/s 实际能用到七成左右，8B 的 4.5 比特模型约 4.5 GB，每秒 9 个 token；上下文到 8K 时 KV（bf16）又多出 1.2 GB，速度掉到 7 个多。所以端侧要同时量化权重和 KV。

`Q4_K` 和 `Q4_0` 平均比特数一样，但 `Q4_K` 用两级缩放，精度接近 5 比特的格式，所以质量排在 `Q4_1` 前面——比特数不是质量的唯一标准。
prefill 是另一回事：1000 个 token 的提示词在 8B 模型上要 16 万亿次运算，手机 NPU 按 20 TFLOPS、四成利用率要 2 秒——这就是端侧"首 token 慢"的原因，也是 NPU 做 prefill、CPU / GPU 做 decode 这种分工的由来。
