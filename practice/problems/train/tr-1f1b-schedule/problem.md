---
title: 生成并模拟 1F1B 流水线调度
chapter: model/pipeline.md
difficulty: 困难
tags: [流水线并行, 1F1B, GPipe, 调度, 气泡]
---
`p` 个 stage、`m` 个 micro-batch 的流水线。操作用元组表示：`("F", i)` 是第 `i` 个 micro-batch 的前向，`("B", i)` 是它的反向。依赖关系：

- stage `s` 的 `("F", i)` 要等 stage `s - 1` 的 `("F", i)` 完成（stage 0 没有这个依赖）；
- stage `s` 的 `("B", i)` 要等本 stage 的 `("F", i)` 和 stage `s + 1` 的 `("B", i)` 都完成（最后一个 stage 只等本 stage 的前向）。

实现：

1. `schedule_1f1b(p, m)`：返回 `p` 个列表，第 `s` 个是 stage `s` 依次执行的操作。1F1B 的规则：先做 `min(p - s - 1, m)` 个前向（预热），然后"一个前向、一个反向"交替，最后把剩下的反向做完；
2. `simulate(p, m, plans, tf=1.0, tb=2.0)`：每个 stage 严格按 `plans[s]` 的顺序执行、同一时刻只做一件事，前向耗时 `tf`、反向耗时 `tb`，每个操作在 stage 空闲且依赖都完成时立即开始。返回全部完成的时刻。如果所有 stage 的下一个操作都在等待、再也推进不了，抛出 `RuntimeError`（死锁）；
3. `peak_activations(plans)`：每个 stage 同时保存激活的 micro-batch 数的最大值（做完前向加一，做完反向减一）。

```python
schedule_1f1b(3, 4)[0]
# [('F', 0), ('F', 1), ('F', 2), ('B', 0), ('F', 3), ('B', 1), ('B', 2), ('B', 3)]
simulate(4, 8, schedule_1f1b(4, 8))      # 33.0 = (m + p - 1) × (tf + tb)
```

<!-- 题解 -->
实测会发现 1F1B 和 GPipe（每个 stage 先做完所有前向、再做所有反向）的总时间完全一样，都是 $(m + p - 1)(t_f + t_b)$，气泡占比 $\frac{p-1}{m+p-1}$；1F1B 的好处在显存：stage $s$ 最多同时保存 $\min(p - s, m)$ 个 micro-batch 的激活，而 GPipe 要保存全部 $m$ 个。
所以 1F1B 允许用更多的 micro-batch 来摊薄气泡而不增加显存。要进一步缩小气泡，就得改变每个 stage 上的模型块（交错调度）或者把反向拆开（零气泡调度）。

模拟器里的死锁检测也有实际意义：真实的流水线用阻塞的 send / recv，调度写错（比如两个相邻 stage 都先等对方发数据）就会卡住，这正是正文里用 `isend` 的原因。
