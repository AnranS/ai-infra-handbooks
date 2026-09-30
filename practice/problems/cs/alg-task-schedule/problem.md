---
title: 任务调度与装箱
chapter: algo/sort-heap-greedy.md
difficulty: 中等
tags: [堆,贪心,调度,装箱]
---
1. `finish_time(tasks, machines)`：`tasks` 是每个任务的耗时，`machines` 台机器各自串行执行。每个任务到来时分给**最早空闲**的机器（用堆），返回全部完成的时刻；
2. `lpt_finish(tasks, machines)`：先把任务按耗时**从大到小**排序再用同样的策略（最长优先，LPT），返回完成时刻；
3. `min_machines(tasks, deadline)`：要在 `deadline` 前用 LPT 策略完成全部任务，至少要几台机器（有任务比 deadline 还长则返回 -1）。

```python
finish_time([5, 3, 8, 2, 7, 1, 6], 3)     # 12
lpt_finish([5, 3, 8, 2, 7, 1, 6], 3)      # 11
min_machines([5, 3, 8], 8)                # 2
```

<!-- 题解 -->
用一个小顶堆存"每台机器的空闲时刻"，每次弹出最小的（最早空闲）、加上任务耗时再压回去。全部完成的时刻就是堆里的最大值。

**LPT（最长优先）**是装箱问题的经典近似算法：先放大的，小任务后面用来填缝。它的完成时刻不超过最优解的 4/3，实践中通常接近最优。推理系统里的对应：把请求按预估长度分组减少批次里的浪费、把专家按负载分配到卡上（EPLB），都是同一类装箱问题。

`min_machines` 从"理论下界"开始试：机器数至少是 `ceil(总耗时 / deadline)`，也至少要能装下最长的任务；从这个下界往上加，直到 LPT 能在 deadline 内完成。这比二分更好写，因为机器数通常不大。
