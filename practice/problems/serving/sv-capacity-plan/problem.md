---
title: 从压测曲线做容量规划
chapter: perf/benchmark.md
difficulty: 简单
tags: [容量规划, SLO, 插值]
---
对一个副本做了多组压测：逐步提高请求速率（QPS），记录 P99 首 token 延迟（TTFT）和 P99 TPOT：

```python
curve = [(1, 0.20, 0.020), (2, 0.25, 0.022), (4, 0.40, 0.028), (6, 0.90, 0.035), (8, 2.50, 0.050)]
#        (qps, ttft_p99, tpot_p99)，按 qps 升序
```

实现：

1. `max_qps_under_slo(curve, ttft_slo, tpot_slo)`：在相邻的测量点之间**线性插值**，求两个指标都不超过 SLO 的最大 QPS。
   假设两个指标都随 QPS 单调不减；最小 QPS 就已经违反 SLO 时返回 `0.0`；最大 QPS 都满足时返回最大 QPS（不外推）；
2. `replicas_needed(target_qps, per_replica_qps, headroom=0.2)`：为了应对突发流量，每个副本只按 `(1 - headroom)` 的容量规划，返回需要的副本数（向上取整；`per_replica_qps` 为 0 时抛出 `ValueError`）；
3. `littles_law_concurrency(qps, mean_latency_s)`：利特尔法则 $L = \lambda W$，系统里平均同时有多少个请求。

<!-- 题解 -->
对每个指标分别求"刚好等于 SLO 的 QPS"：找到第一个超过 SLO 的点 $i$，在 $(q_{i-1}, v_{i-1})$ 和 $(q_i, v_i)$ 之间解 $v = \text{slo}$，
$q = q_{i-1} + (\text{slo} - v_{i-1}) \frac{q_i - q_{i-1}}{v_i - v_{i-1}}$。两个指标的结果取较小值。

容量规划的关键是**在 SLO 下**的吞吐，而不是极限吞吐：上面的曲线里，延迟在 6 QPS 之后急剧上升（排队），极限吞吐虽然更高，但用户体验不可接受。
利特尔法则常用来反推需要的并发数（`max_num_seqs`）和 KV 显存。
