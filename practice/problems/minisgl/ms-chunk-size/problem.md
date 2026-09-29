---
title: 分块 prefill 的块大小怎么选
chapter: schedule/chunked-prefill.md
difficulty: 中等
tags: [分块 prefill, 延迟, 模拟]
---
一个长提示词到达时，如果一次性 prefill，这一轮会非常慢，所有正在 decode 的请求都要等它（TPOT 出现尖刺）；分块 prefill 把它切成若干块，每轮只算一块，和 decode 请求**分别**占用轮次（mini-sglang 一轮只做一种：有 prefill 就先做 prefill）。

用一个简单的耗时模型：一轮算 `t` 个 token 的时间是 `a + b * t` 毫秒。场景：有 `n_decode` 个请求在持续 decode；一个长度为 `L` 的提示词到达。调度规则：

- 按块大小 `C` 把提示词切成 `ceil(L / C)` 块；
- 轮次交替进行：一轮 prefill（算一块）、一轮 decode（所有 decode 请求各 1 个 token），直到 prefill 完成（最后一块之后也紧跟一轮 decode）；
- 从提示词到达（第一轮 prefill 开始）算起。

实现：

1. `simulate(L, C, n_decode, a, b)`：返回 `(ttft, max_gap)`：`ttft` 是最后一块 prefill 完成的时刻；`max_gap` 是 decode 请求**相邻两个 token 之间**的最长间隔 = 一轮 prefill + 一轮 decode 的时间（取最大的那一块）；
2. `best_chunk(L, n_decode, a, b, tpot_slo, candidates)`：在候选块大小里，选满足 `max_gap <= tpot_slo` 的块里 **TTFT 最小**的（相同时取更大的块）；都不满足时返回 `None`。

<!-- 题解 -->
块大小是 TTFT 和 TPOT 之间的权衡：块越小，decode 被打断的时间越短（`max_gap` 小），但 prefill 被切成更多轮、每轮都有固定开销 `a`，还要和 decode 轮交替，TTFT 变长。

```text
blocks = [C] * (L // C) + ([L % C] if L % C else [])
ttft = Σ (a + b·c_i) + (块数 - 1) · (a + b·n_decode)      # 最后一块之后的 decode 轮不计入 TTFT
max_gap = a + b·max(c_i) + a + b·n_decode
```

SGLang 正式版支持"混合批"（一轮里 prefill 块和 decode 一起算），固定开销只付一次，这是 mini-sglang 没有实现的优化。
