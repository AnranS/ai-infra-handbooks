---
title: 依赖链与指令级并行
chapter: arch/cpu.md
difficulty: 中等
tags: [流水线, 依赖链, ILP, 估算]
---
一个乱序 CPU 每周期最多发射 `width` 条指令，同一种指令每周期最多发射 `throughput` 条，一条指令的结果要 `latency` 个周期后才能被依赖它的指令使用。实现两个函数：

1. `run_cycles(n_ops, accumulators, latency, throughput, width)`：一个循环把 `n_ops` 个数加起来，用 `accumulators` 个互不依赖的累加器轮流累加（第 `i` 次加法用第 `i % accumulators` 个累加器，依赖同一个累加器的上一次加法）。返回全部加法**完成**所需的周期数：每条加法在"它依赖的结果已经产出"且"这个周期还有发射名额"时发射，第 `c` 周期发射的加法在第 `c + latency` 周期产出结果；
2. `min_accumulators(latency, throughput)`：要让发射端不因为等待结果而空闲，至少需要几个累加器。

```python
run_cycles(8, 1, 4, 2, 4)      # 32：每次加法都要等上一次，8 × 4
run_cycles(8, 8, 4, 2, 4)      # 7：每周期发 2 条，第 3 周期发完，最后一条 4 周期后产出
min_accumulators(4, 2)         # 8：延迟 × 吞吐
```

<!-- 题解 -->
`min_accumulators` 就是 Little 定律的一个特例：要让每个周期都能发出 `throughput` 条加法，而每条加法要 `latency` 个周期才产出，同时"在飞"的加法必须有 `latency × throughput` 条，它们必须互不依赖，也就是需要这么多个累加器。

`run_cycles` 逐周期模拟：维护每个累加器"下一次可用的周期"（初始为 0）和这个周期已经发射的条数；按顺序取下一条加法，如果它的累加器已经可用且名额没用完就发射，把这个累加器的可用周期设为 `cycle + latency`；否则等下一个周期。完成周期是最后一条加法的产出周期。

把 `accumulators` 从 1 加到 8，64 次加法的周期数从 256 降到 35，正好对应本章实测的 1.33 ns → 0.17 ns。超过 `latency × throughput` 之后就不再变快：瓶颈从依赖链换成了发射带宽。
