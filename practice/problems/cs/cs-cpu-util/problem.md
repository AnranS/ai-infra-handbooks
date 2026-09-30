---
title: 从 /proc/stat 算 CPU 使用率
chapter: os/perf-tools.md
difficulty: 简单
tags: [性能分析, /proc, CPU 使用率]
---
`top`、`mpstat` 的 CPU 使用率都是从 `/proc/stat` 的两次快照算出来的。`/proc/stat` 里每个 CPU 一行，形如 `cpu3 user nice system idle iowait irq softirq steal guest guest_nice`，都是从开机起的累计值（单位是时钟滴答）；第一行 `cpu`（后面没有编号）是所有 CPU 的合计，忽略它。实现 `cpu_usage(before, after)`，两个参数是两次读到的文件内容，返回：

- `"per_cpu"`：字典 `CPU 编号 → 使用率（百分比，保留 1 位小数）`。使用率 = 100 × (总时间的增量 − idle 的增量 − iowait 的增量) / 总时间的增量，总时间是这一行前 8 个字段（user 到 steal）的和；增量为 0 的 CPU 记为 0.0；
- `"busiest"`：使用率最高的 CPU 编号（一样高时取编号小的）；
- `"single_thread_bottleneck"`：最忙的 CPU 使用率 ≥ 90，并且其余 CPU 都 ≤ 30 时为 `True`——这通常是一个单线程的热点（比如推理引擎的 Python 调度循环）。

```python
before = "cpu  0 0 0 0 0 0 0 0\ncpu0 100 0 50 850 0 0 0 0\ncpu1 10 0 10 980 0 0 0 0\n"
after  = "cpu  0 0 0 0 0 0 0 0\ncpu0 190 0 58 852 0 0 0 0\ncpu1 12 0 12 1076 0 0 0 0\n"
cpu_usage(before, after)
# {"per_cpu": {0: 98.0, 1: 4.0}, "busiest": 0, "single_thread_bottleneck": True}
```

<!-- 题解 -->
使用率是"非空闲时间占总时间的比例"，而且必须用两次快照的**差值**：文件里是开机以来的累计值，直接用会得到开机以来的平均使用率。iowait 要算成空闲：它表示 CPU 没事做、只是恰好有任务在等 I/O，这段时间别的任务随时可以用这个 CPU——把它算成"忙"会高估使用率，也会误导你去优化 CPU。steal 计入总时间（虚拟机里被宿主机拿走的时间），但不算空闲。

一个核 100%、其余很闲，是单线程瓶颈的典型形状：整机平均使用率可能才 5%，看平均值会以为 CPU 很充裕。所以排查时用 `mpstat -P ALL` 看每个核，而不是只看 `top` 顶部的总数。
