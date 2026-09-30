---
title: 估算：KV 块与并发数
chapter: algo/sort-heap-greedy.md
difficulty: 简单
tags: [估算,KV Cache,分页]
---
分页 KV Cache 按固定大小的块分配。实现：

1. `blocks_for(tokens, block_size)`：一个请求需要几个块（不满一块也占一整块）；
2. `waste_tokens(tokens, block_size)`：最后一块里浪费的槽位数；
3. `max_concurrent(total_blocks, tokens_each, block_size)`：给定总块数，能同时服务几个这样的请求；
4. `block_bytes(block_size, n_layers, n_kv_heads, head_dim, dtype_bytes=2)`：一个块占多少字节（K 和 V 各一份）。

```python
blocks_for(100, 16)                              # 7
waste_tokens(100, 16)                            # 12
max_concurrent(1000, 2048, 16)                   # 7
block_bytes(16, 32, 8, 128, 2)                   # 2097152
```

<!-- 题解 -->
块大小是一组权衡：**块小**则内部碎片少（`waste_tokens` 小）、前缀复用的粒度细，但块表更长、每次访存的元数据开销更大；**块大**则相反。vLLM 默认 16，SGLang 默认 1（token 级），各有取舍。

一个块的大小 = `块内 token 数 × 层数 × KV 头数 × 头维度 × 精度字节数 × 2`（K 和 V）。这个公式是所有 KV 显存估算的基础：32 层、8 个 KV 头（GQA）、头维度 128、BF16 时，每个 token 占 128 KB，一个 16 token 的块就是 2 MB。

由此可以反推并发：显存里留给 KV 的部分 ÷ 每块字节数 = 总块数，再除以每个请求需要的块数。长上下文会让这个数迅速变小——这正是 MLA、KV 量化、KV 卸载要解决的问题（见 [KV Cache 与两阶段推理](llm://inference/kv-cache/)）。
