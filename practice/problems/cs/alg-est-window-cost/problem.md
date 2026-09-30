---
title: 估算：分块 prefill 的窗口代价
chapter: algo/array-string.md
difficulty: 简单
tags: [估算,滑动窗口,分块 prefill]
---
分块 prefill 把一次长 prefill 切成若干块，每块不超过 `chunk` 个 token，和 decode 请求混在同一批里跑。注意力的计算量与"这一块的 token 数 × 它能看到的上下文长度"成正比。实现：

1. `chunks_of(prompt_len, chunk)`：切成几块（最后一块可以不满）；
2. `attention_tokens(prompt_len, chunk)`：所有块的"块内 token 数 × 可见上下文长度"之和。第 i 块（从 0 开始）有 `c` 个 token，它要对"前面所有块的 token"各算一次（`c * 前缀长度`），块内部是因果注意力（`c * (c + 1) / 2`）；
3. `overhead(prompt_len, chunk)`：分块后的注意力代价相对不分块（`chunk = prompt_len`）的倍数。

```python
chunks_of(5000, 2048)                    # 3
attention_tokens(4, 4)                   # 10：4 个 token 的因果注意力
round(overhead(8192, 512), 3)            # 1.0：分块不改变注意力总量
round(overhead(8192, 8192), 3)           # 1.0
```

<!-- 题解 -->
这道题想说明一件常被误解的事：**分块 prefill 本身不增加注意力的计算量**。每个 token 要看的上下文（它自己和它前面的所有 token）是固定的，怎么切块都一样，总量都是 `n(n+1)/2`，所以 `overhead` 恒等于 1。

真正的代价在别处：每块都要重新读一遍前面所有块的 KV（访存变多）、每块都要启动一轮 kernel（固定开销变多）、块太小则矩阵乘的形状变差（算力利用率下降）。所以块大小是在"和 decode 混批的延迟收益"与"这些固定开销"之间折中，典型值 512～2048 个 token。

真实系统里还要考虑前缀缓存命中（跳过已缓存的块）、以及 chunk 与 CUDA Graph 的批次分桶对齐，见[分块 prefill](minisgl://schedule/chunked-prefill/)。
