---
title: 部署前的显存规划
chapter: perf/quantization-deploy.md
difficulty: 简单
tags: [估算, 显存, 量化, KV Cache]
---
上线前要回答："这张卡能放下这个模型吗？最多能同时服务多少个请求？"实现 `plan(cfg, gpu_mem_gb, weight_dtype, kv_dtype, context_len, util=0.9, activation_gb=2.0)`：

- `cfg`：`{"params": 参数量, "n_layers":, "n_kv_heads":, "head_dim":}`；
- 权重字节数：`bf16`/`fp16` 每参数 2 字节，`fp8`/`int8` 1 字节，`int4` 0.5 字节**另加每 128 个参数一组的 fp16 scale**（每组 2 字节）；
- KV Cache 每个 token 的字节数：`2 × n_layers × n_kv_heads × head_dim × KV 元素字节数`，`kv_dtype` 为 `bf16`/`fp16`（2）或 `fp8`（1）；
- 可用显存：`gpu_mem_gb × 2^30 × util`，减去权重和激活预留（`activation_gb × 2^30`）后全部给 KV Cache；

返回字典：`weight_gb`、`kv_bytes_per_token`、`kv_tokens`（能放下的 token 数，向下取整；不够时为 0）、`max_seqs`（按每个请求 `context_len` 个 token 算，向下取整）、`fits`（KV Cache 至少能放下一个完整请求时为 `True`）。
`weight_gb` 按 $2^{30}$ 字节为 1 GB。未知的 dtype 抛出 `ValueError`。

```python
plan(llama_8b, 80, "bf16", "bf16", 8192)   # 权重约 15 GB，每 token 128 KB，约 45 万个 token，能同时服务 55 个 8K 上下文的请求
```

<!-- 题解 -->
`int4` 每组 128 个参数占 `128 × 0.5 + 2 = 66` 字节，平均每参数 0.515625 字节。
KV Cache 常常才是容量的决定因素：8B 模型权重只占 15 GB，剩下的约 55 GB 全给 KV 也只够 55 个 8K 请求；
把 KV 换成 FP8，容量翻倍——这比把权重从 bf16 量化到 int4 带来的收益还大（在长上下文、高并发时）。
