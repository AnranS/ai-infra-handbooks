---
title: 跳跃游戏
chapter: algo/sort-heap-greedy.md
difficulty: 中等
tags: [贪心,跳跃]
---
`nums[i]` 表示在位置 i 最多能往前跳几步。

1. `can_jump(nums)`：能否从第一个位置跳到最后一个；
2. `min_jumps(nums)`：最少跳几次到达最后一个位置（保证能到达；只有一个元素返回 0）。

```python
can_jump([2, 3, 1, 1, 4])      # True
can_jump([3, 2, 1, 0, 4])      # False
min_jumps([2, 3, 1, 1, 4])     # 2
```

<!-- 题解 -->
`can_jump` 维护"目前能到达的最远位置" `reach`：扫描时如果当前下标已经超过 `reach`，说明断了；否则更新 `reach = max(reach, i + nums[i])`。O(n)、O(1) 空间。

`min_jumps` 是分层的贪心（本质是 BFS）：维护当前这一跳能覆盖的边界 `end` 和下一跳能到的最远处 `farthest`。走到 `end` 时跳数加一、把 `end` 更新成 `farthest`。注意循环到 `n - 1` 就停——否则最后一步会多算一次。

两题都体现了贪心的特点：不需要知道"具体怎么跳"，只要维护一个"可达范围"。类似的题：加油站（能否绕一圈）、视频拼接、最少区间覆盖。
