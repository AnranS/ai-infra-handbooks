---
title: 确定性推理：固定长度的 split-KV 与按种子采样
chapter: topics/deterministic.md
difficulty: 中等
tags: [确定性推理, batch 不变性, split-KV, 采样]
requires: [numpy]
---
decode 注意力的 split-KV：把一个请求的 KV 切成几段分别算，再用 log-sum-exp 合并。模板里的 `heuristic_batch_decode` 按 batch 大小决定切几段（请求少就多切几段，用满 SM），于是同一个请求的结果取决于它和谁在一个 batch 里。实现 batch 不变的版本（全部用 `float32` 计算）：

1. `decode_attention(q, K, V, split_len)`：一个请求、一个头。`q` 形状 `(d,)`，`K` 是 `(n, d)`，`V` 是 `(n, dv)`。KV 从头开始每 `split_len` 个一段（最后一段可以不满），每段算出分数 `K_c @ q / sqrt(d)`、这一段的 softmax 加权和 `o_c` 以及 `lse_c = log Σ exp(分数)`，再**从左到右**依次合并：
   $$\text{lse} = \log(e^{\text{lse}_1} + e^{\text{lse}_2}),\quad o = o_1 e^{\text{lse}_1 - \text{lse}} + o_2 e^{\text{lse}_2 - \text{lse}}$$
2. `batch_decode(requests, split_len=256)`：`requests` 是 `(q, K, V)` 的列表，返回每个请求的输出。无论 batch 里有几个、有哪些请求，同一个请求的输出都要**逐位相同**；
3. `sample(probs, seed, position)`：从分布 `probs` 里采一个 token。随机数只能由 `(seed, position)` 决定（模板给了 `splitmix64`，把它们变成 `[0, 1)` 里的一个数，再按累积概率找到第一个超过它的 token），不能用全局的随机数流——否则 batch 里请求的顺序一变，采到的 token 就变了。

<!-- 题解 -->
batch 不变的关键是"切多长"而不是"切几段"固定：每段 `split_len` 个 token，段数只由这个请求自己的长度决定，合并的顺序也固定，于是它的浮点运算序列与 batch 无关。SGLang 的确定性模式正是这样：FlashInfer 的 decode 按 2048 个 token 切。代价是短 batch 时并行度不够。

合并时要先减去较大的那个 lse 再取指数（`np.logaddexp` 已经这样做了），否则分数大时会溢出。

采样用的是**计数器式**随机数：`u = f(seed, position)`，同一个请求在同一个位置总是得到同一个 `u`，与 batch 里的其他请求、请求的排列顺序都无关。SGLang 的 `multinomial_with_seed`、vLLM 按请求设置的 `seed` 都是这个思路；JAX 的随机数也是计数器式的。
