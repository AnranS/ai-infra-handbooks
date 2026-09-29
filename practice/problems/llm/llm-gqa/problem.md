---
title: GQA 注意力与 KV Cache 显存
chapter: transformer/attention-variants.md
difficulty: 中等
tags: [GQA, MQA, KV Cache]
---
分组查询注意力（GQA）让多个 query 头共享一组 key/value 头，KV Cache 因此成倍缩小。

实现：

1. `repeat_kv(kv, n_rep)`：`kv` 形状 `(S, H_kv, d_h)`，把每个 KV 头重复 `n_rep` 次，得到 `(S, H_kv · n_rep, d_h)`，
   顺序是 `[头0, 头0, …, 头1, 头1, …]`（第 $h$ 个 query 头对应第 $h // n_{rep}$ 个 KV 头）；
2. `gqa_attention(q, k, v)`：`q` 形状 `(T, H_q, d_h)`，`k`、`v` 形状 `(S, H_kv, d_h)`，$S \ge T$，$H_q$ 是 $H_{kv}$ 的整数倍。
   这 $T$ 个 query 是整个序列的**最后** $T$ 个位置（前面 $S - T$ 个位置的 KV 来自缓存），因果掩码按"右下角对齐"：第 $i$ 个 query 的绝对位置是 $S - T + i$，只能看到位置 $\le S - T + i$ 的 key。返回 `(T, H_q, d_h)`；
3. `kv_cache_bytes(n_layers, n_kv_heads, head_dim, n_tokens, bytes_per_elem=2)`：KV Cache 占用的字节数（K 和 V 都要存）。

```python
kv_cache_bytes(28, 8, 128, 32768)      # Qwen3-0.6B 存 32K 个 token：3,758,096,384 字节 = 3.5 GiB
```

<!-- 题解 -->
`repeat_kv` 就是 `np.repeat(kv, n_rep, axis=1)`（注意不是 `np.tile`，后者的顺序是 `[头0, 头1, …, 头0, 头1, …]`）。

右下角对齐的掩码：`allowed[i, j] = j <= S - T + i`。prefill 时 $S = T$，就是普通的下三角；decode 时 $T = 1$，唯一的 query 能看到全部 $S$ 个 key。
实际的 kernel 不会真的复制 KV：它直接用 `h // n_rep` 去索引 KV 头，省下显存和带宽。
