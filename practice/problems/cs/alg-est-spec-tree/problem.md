---
title: 估算：投机解码的接受长度
chapter: algo/dp-backtrack.md
difficulty: 中等
tags: [估算,投机解码,期望]
---
投机解码：草稿模型一次生成 `k` 个 token，目标模型一次前向验证它们，接受最长的正确前缀（第一个错的 token 之后全部作废，但会用目标模型的输出补上一个）。设每个 token 被接受的概率都是 `p`（互相独立）。实现：

1. `accept_expect(p, k)`：一次验证**期望产出的 token 数**（接受的前缀长度 + 补上的那一个），等于 `sum(p^i for i in 0..k)`；
2. `speedup(p, k, draft_cost, verify_cost)`：相对不用投机解码的加速比。一轮的成本是 `k * draft_cost + verify_cost`，产出是 `accept_expect(p, k)`；不用投机时每个 token 的成本是 `verify_cost`；
3. `best_k(p, draft_cost, verify_cost, max_k=16)`：使加速比最大的 `k`（并列取小的）。

```python
round(accept_expect(0.8, 4), 3)              # 3.362
round(speedup(0.8, 4, 0.1, 1.0), 3)          # 2.401
best_k(0.8, 0.1, 1.0)                        # 6
best_k(0.3, 0.1, 1.0)                        # 1
```

<!-- 题解 -->
期望接受长度是等比数列求和：第 0 个 token（目标模型自己出的）一定产出，第 i 个草稿 token 被接受的概率是 `p^i`，所以期望是 `1 + p + p² + … + p^k`。

加速比 = 期望产出 ÷ 一轮成本 × 不用投机时每 token 的成本。草稿越长，期望产出越多，但成本线性增加而收益指数衰减，所以存在一个**最优的 k**：接受率高（p 大）时 k 可以大，接受率低时 k 要小，否则白算。

真实系统里 `p` 不是常数（越往后越难猜），草稿也不是链而是**树**（一次验证多条分支，接受率更高但验证成本也更高），还要考虑 batch 变大后验证的开销。这道题的模型足够解释"为什么 k 通常取 3～5"以及"为什么接受率低的场景干脆别开投机解码"。见[投机解码进阶](serving://topics/speculative/)。
