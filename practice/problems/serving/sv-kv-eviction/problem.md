---
title: 长上下文的 KV 淘汰：StreamingLLM 与 H2O
chapter: topics/long-context.md
difficulty: 中等
tags: [KV 淘汰, attention sink, 长上下文]
---
KV Cache 放不下时只保留一部分 token。两种经典策略：

- **StreamingLLM**：保留开头的 `sink` 个 token（注意力汇聚点，删掉它们模型会崩）和最近的 `window` 个 token；
- **H2O**（Heavy-Hitter Oracle）：保留最近的 `recent` 个 token，再加上**累计注意力分数最高**的若干个"重要" token，总数不超过 `budget`。

实现：

1. `streaming_keep(n, sink, window)`：序列长度为 `n` 时保留的位置（升序列表）。`n <= sink + window` 时全部保留；
2. `h2o_keep(acc, budget, recent)`：`acc` 是长度为 `n` 的累计注意力分数。`n <= budget` 时全部保留；否则保留最后 `recent` 个，再在其余位置里按分数从高到低（同分取位置靠前的）补足到 `budget` 个。返回升序列表；
3. `simulate_h2o(attn_steps, budget, recent)`：模拟 decode 过程。一开始缓存为空；第 `t` 步（`t = 0, 1, ...`）新 token（绝对位置 `t`）进入缓存，
   `attn_steps[t]` 是一个字典 `{绝对位置: 注意力权重}`，给出这一步的 query 对**当前缓存里每个位置（含自己）**的注意力；
   把它累加到各位置的累计分数上，然后如果缓存超过 `budget` 个，用 `h2o_keep` 的规则淘汰（被淘汰的位置从此消失，分数也丢掉）。
   返回最后缓存里的绝对位置（升序）。

<!-- 题解 -->
`h2o_keep`：`recent_set = range(n - recent, n)`，其余位置 `sorted(others, key=lambda i: (-acc[i], i))[:budget - recent]`。

模拟时维护 `cache`（绝对位置列表）和 `score` 字典；每步先追加新位置、累加分数，再淘汰。注意 `h2o_keep` 的下标是缓存里的**相对**下标，需要映射回绝对位置。
StreamingLLM 还会让位置编码使用"缓存内的相对位置"（否则长度超过训练长度后 RoPE 外推会出问题），这里只做选择。
