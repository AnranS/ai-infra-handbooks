---
title: 延迟指标：TTFT、TPOT、ITL 与 goodput
chapter: engine/overview.md
difficulty: 简单
tags: [压测, SLO, TTFT, TPOT]
---
压测工具记录每个请求的到达时刻和每个输出 token 发回的时刻（秒）。按压测报告的通用定义实现：

1. `request_metrics(arrival, token_times)`：`token_times` 至少有一个元素、非递减。返回字典：`ttft`（首 token 延迟 = 第一个 token 的时刻 − 到达时刻）、`tpot`（(最后一个 − 第一个) / (token 数 − 1)，只有一个 token 时为 0）、`itl`（相邻 token 的间隔列表）、`e2e`（最后一个 token 的时刻 − 到达时刻）；
2. `percentile(values, p)`：最近秩（nearest-rank）分位数：排序后取第 `ceil(p / 100 × n)` 个（从 1 数起，至少取第 1 个、最多取第 n 个）；
3. `summarize(requests, ttft_slo, tpot_slo)`：`requests` 是 `[(arrival, token_times), ...]`。返回字典：`ttft_p50`、`ttft_p99`、`tpot_p50`、`tpot_p99`、`itl_p99`（所有请求的所有间隔放在一起算）、`slo_ok`（TTFT 和 TPOT 都不超过 SLO 的请求占比）、`goodput`（满足 SLO 的请求数 ÷ 时长，时长 = 最后一个 token 的时刻 − 最早的到达时刻）。

```python
m = request_metrics(0.0, [0.5, 0.52, 0.54, 0.60])
m["ttft"], m["tpot"]          # (0.5, 0.0333...)：TPOT 是 (0.60 − 0.5) / 3
percentile([5, 1, 4, 2, 3], 50)   # 3
```

<!-- 题解 -->
TTFT 包括排队、prefill 和各种 CPU 处理；TPOT 是一个请求首 token 之后的平均速度；ITL 是每一个间隔。TPOT 是平均值，会把卡顿平均掉：一次长 prefill 插进来，某个请求的一个间隔变成 300 ms，它的 TPOT 可能只多几毫秒，但 ITL 的 P99 会暴露它——所以压测报告要同时看 TPOT 和 ITL 的尾部。

分位数用最近秩的定义，P99 就是"99% 的请求都不超过的那个值"；样本少时它就是最大值附近的那个。goodput 只数满足 SLO 的请求：负载越过拐点之后，吞吐还会涨一点，但满足 SLO 的请求比例急剧下降，goodput 反而崩溃——这才是容量规划要看的数。
