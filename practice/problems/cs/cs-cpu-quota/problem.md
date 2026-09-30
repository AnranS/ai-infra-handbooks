---
title: 容器的 CPU 配额与节流
chapter: os/containers.md
difficulty: 中等
tags: [容器, cgroup, 节流, 延迟]
---
Kubernetes 的 `limits.cpu` 在 cgroup 里变成 CFS 带宽控制：每个周期（`period_ms`）里，整个容器最多用 `quota_ms` 毫秒的 CPU 时间，用完就被挂起到下一个周期开始。实现：

1. `usable_cpus(affinity, quota_us, period_us)`：容器真正能用的 CPU 数。`affinity` 是允许运行的 CPU 数；`quota_us` 为 `-1` 表示不限，否则是 `min(affinity, quota_us / period_us)`；
2. `finish_ms(threads, work_ms, quota_ms, period_ms, cores)`：`threads` 个线程同时开工，每个要 `work_ms` 毫秒的 CPU 时间；机器上最多 `cores` 个线程能同时运行（多出来的轮流，这里简化成大家平均分配 `cores` 个核）；`quota_ms` 为 `None` 表示不限配额。返回全部完成的时刻（毫秒）。周期从 0 开始，第 k 个周期是 `[k·period, (k+1)·period)`。

```python
finish_ms(8, 40, 200, 100, cores=64)      # 115.0：前 25 ms 就用光了 200 ms 配额，等到 100 ms 再跑 15 ms
finish_ms(8, 40, None, 100, cores=2)      # 160.0：没有配额限制，但只有 2 个核
usable_cpus(32, 200000, 100000)           # 2.0
```

<!-- 题解 -->
模拟按事件推进：在一个周期内，同时运行的线程数是 `min(剩下的线程数, cores)`，配额的消耗速度是这个数（每毫秒消耗这么多毫秒的 CPU 时间）。下一个事件是三者中最早的：某个线程做完、配额用光、周期结束。配额用光时直接跳到下一个周期的开始，配额重置。

结论和直觉不太一样：平均 CPU 使用率远没到上限，延迟却可能翻好几倍——8 个线程的突发在 25 ms 内用光了 100 ms 周期的配额，剩下 75 ms 全在等。所以延迟敏感的服务常常不设 CPU limits，或者用独占核，并按配额限制线程数。`usable_cpus` 说明了为什么不能按 `os.cpu_count()` 开线程：它既不知道 cpuset，也不知道配额。
