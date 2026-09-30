---
title: 买卖股票
chapter: algo/dp-backtrack.md
difficulty: 中等
tags: [动态规划,状态机,贪心]
---
`prices[i]` 是第 i 天的价格。

1. `max_profit_once(prices)`：只能买卖**一次**的最大利润（不交易则为 0）；
2. `max_profit_many(prices)`：可以买卖**任意多次**（同一天可以先卖后买）的最大利润；
3. `max_profit_cooldown(prices)`：任意多次，但卖出后**第二天不能买**（冷冻期一天）。

```python
max_profit_once([7, 1, 5, 3, 6, 4])      # 5
max_profit_many([7, 1, 5, 3, 6, 4])      # 7
max_profit_cooldown([1, 2, 3, 0, 2])     # 3
```

<!-- 题解 -->
三道题是同一个"状态机 DP"的不同版本，状态是"今天结束时手里有没有股票"：

- **只买一次**：维护"到今天为止的最低价"，答案是 `max(今天价格 - 最低价)`。等价于 DP 里"买入状态只能从 0 转移"；
- **任意多次**：只要明天比今天贵就今天买明天卖，答案是所有上涨段的和（贪心），也可以写成 `hold = max(hold, cash - price)`、`cash = max(cash, hold + price)`；
- **带冷冻期**：卖出后要等一天，所以买入要从"前天的 cash"转移，多存一个 `prev_cash` 即可。

写状态机 DP 的要点：先把状态画出来（持有 / 不持有 / 冷冻），再写每个状态的转移。买卖次数有限制（最多 k 次）时状态再加一维。这类题是"状态设计"的最佳练习。
