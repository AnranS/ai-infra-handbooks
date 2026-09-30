---
title: 估算：前缀缓存能省多少
chapter: algo/tree-graph.md
difficulty: 简单
tags: [估算,前缀缓存,Radix Cache]
---
估算前缀缓存带来的收益。一次 prefill 的耗时按"要算的 token 数 × 每个 token 的耗时"计。实现：

1. `matched_len(cached, request)`：两个 token 序列（元组）的公共前缀长度；
2. `prefill_ms(tokens, per_token_ms=0.15)`：算这么多 token 要多久；
3. `saved_ms(cached, request, per_token_ms=0.15)`：命中前缀省下的时间；
4. `hit_rate_needed(ttft_target_ms, prompt_len, per_token_ms=0.15, overhead_ms=20)`：给定 TTFT 目标，前缀命中率（命中的 token 数 ÷ prompt 长度）至少要多少。`overhead_ms` 是排队和其他固定开销；已经满足则返回 0.0，无论如何都达不到返回 1.0。

```python
matched_len((1, 2, 3, 4), (1, 2, 9))          # 2
round(prefill_ms(4096), 1)                    # 614.4
round(saved_ms((1, 2, 3), (1, 2, 9)), 2)      # 0.3
round(hit_rate_needed(200, 4096), 3)          # 0.707
```

<!-- 题解 -->
前缀缓存的收益完全取决于**公共前缀有多长**：命中的部分不用重算 prefill，直接复用 KV。所以工程上最有效的做法是把不变的内容（系统提示词、few-shot 例子、工具定义）放在最前面，把变化的内容（用户输入）放在最后。

`hit_rate_needed` 是一个实用的反推：TTFT 目标 200 ms、prompt 4096 个 token、每 token 0.15 ms、固定开销 20 ms 时，需要至少 70.7% 的前缀命中率。达不到就得换方案——缩短 prompt、加卡、或者用 PD 分离把 prefill 挪到专门的实例上。

真实系统还要考虑：块粒度（vLLM 按 16 个 token 一块，不足一块的尾巴不能复用）、缓存容量与淘汰、以及路由能不能把同前缀的请求送到同一个实例（见[一致性哈希](../dist/hash-shard.md)）。
