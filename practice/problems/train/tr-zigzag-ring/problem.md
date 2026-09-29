---
title: 之字形切分的 Ring Attention
chapter: model/context.md
difficulty: 困难
tags: [上下文并行, Ring Attention, 因果注意力, 负载均衡]
requires: [numpy]
---
`P` 个 rank 做因果注意力的上下文并行。序列长度 `S` 能被 `2P` 整除，切成 `2P` 块（每块 `c = S / 2P` 个 token）。

1. `zigzag_order(P)`：rank `r` 负责第 `r` 块和第 `2P - 1 - r` 块（一前一后），返回每个 rank 的两个块号 `[[0, 2P-1], [1, 2P-2], ...]`；
2. `ring_attention_zigzag(q, k, v, P)`：模拟 Ring Attention。每个 rank 保留自己两块的 query；KV 也按同样的方式分好，沿环传递：第 `i` 步（`i = 0..P-1`）时 rank `r` 手里是 rank `(r - i) mod P` 的两块 KV。每一步，rank 用自己的每个 query 块去和手里的每个 KV 块算注意力：
   - KV 块在 query 块之前：整块都可见；
   - 同一块：块内因果掩码；
   - KV 块在 query 块之后：整块跳过。

   各部分用 log-sum-exp 合并（和 FlashAttention 的 online softmax 同一个公式），最后按原来的顺序拼回 `(S, dv)` 的输出。分数的缩放是 `1 / sqrt(d)`。

   返回 `(out, work)`：`work[i][r]` 是第 `i` 步 rank `r` 实际计算的 (query, key) 对数——整块可见算 `c²`，块内因果只算可见的 `c(c + 1) / 2`，跳过的不算。

之字形切分让每一步每个 rank 的计算量都相同：最慢的 rank 不再拖累整个环。

<!-- 题解 -->
合并公式：已有 `(o₁, lse₁)`，新来一段 `(o₂, lse₂)`，`lse = logaddexp(lse₁, lse₂)`，`o = o₁·e^{lse₁ - lse} + o₂·e^{lse₂ - lse}`（逐行）。一个 query 块第一次遇到可见的 KV 块时直接用它的结果初始化。

为什么均衡：第 0 步各 rank 算的是自己的两块，都是"两个对角块 + 一个整块"；之后每一步，来自 rank `j` 的两块 KV 相对于 rank `r` 的两块 query，不论 `j < r` 还是 `j > r`，都恰好有两个块对完全可见、两个完全跳过，工作量都是 `2c²`。按顺序切（rank `r` 拿第 `2r`、`2r+1` 块）时，rank 0 几乎不用算、rank `P-1` 算得最多，总时间由最慢的 rank 决定。

真实实现里，块内因果的部分交给支持因果掩码的 kernel，跳过的块根本不发起计算；Megatron 的上下文并行、Llama 3 的长上下文训练用的都是这种之字形（或条带式）切分。
