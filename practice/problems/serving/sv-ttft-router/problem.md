---
title: 按预计 TTFT 路由与提前拒绝
chapter: frontier/disagg-sched.md
difficulty: 中等
tags: [路由, 前缀缓存, 过载控制, PD 分离]
---
实现一个 prefill 实例的全局路由器 `Router(n, speed, slo)`：`n` 个实例，每个实例每秒能 prefill `speed` 个 token，TTFT 目标 `slo` 秒。每个实例维护：

- `busy[i]`：它的队列排到什么时刻（初始为 0）；
- 一个已缓存前缀的集合（用任意可哈希的"前缀键"表示，初始为空）。

`route(t, prefix_keys, length)`：请求在时刻 `t` 到达，`prefix_keys` 是它从头开始的各个前缀块的键（列表），`length` 是它的总 token 数，每个块 `block` 个 token（构造时的参数，默认 64）。

1. 对每个实例计算命中的 token 数：从头开始连续命中的块数 × `block`；
2. 预计 TTFT = `max(busy[i] - t, 0) + (length - 命中) / speed`；
3. 选预计 TTFT 最小的实例（相同时选编号小的）；如果最小值都超过 `slo`，**拒绝**，返回 `None`，不改变任何状态；
4. 否则把请求排进这个实例：`busy[i] = max(busy[i], t) + (length - 命中) / speed`，把 `prefix_keys` 全部加入它的缓存，返回 `(实例编号, 命中 token 数, 实际 TTFT)`，其中实际 TTFT = 新的 `busy[i] - t`。

```python
r = Router(2, speed=1000, slo=1.0, block=64)
r.route(0.0, ["a", "b"], 200)       # (0, 0, 0.2)
r.route(0.0, ["a", "b"], 200)       # (1, 0, 0.2)  实例 0 能命中 128 个 token 但要排队，预计 0.272 s；实例 1 预计 0.2 s
r.route(0.5, ["a", "b", "c"], 300)  # (0, 128, 0.172)
```

<!-- 题解 -->
这就是正文模拟里"预计 TTFT 最小 + 提前拒绝"的策略：命中越多、要算的越少；排队越长、等得越久，两者折算成同一个单位（秒）再比较。提前拒绝在请求消耗任何算力之前就做出判断，让注定超时的请求不拖累后面的请求。
真实的路由器看不到实例真实的缓存状态（实例会自己淘汰），这里的集合只是路由器的近似索引。
