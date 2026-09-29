---
title: TTFT、TPOT 与 SLO 下的有效吞吐
chapter: inference/serving.md
difficulty: 简单
tags: [指标, 分位数, SLO]
---
压测结束后，每个请求有这样一条记录（时间单位：秒）：

```python
{"arrival": 0.00, "first_token": 0.35, "finish": 2.35, "output_tokens": 101}
```

实现 `summarize(records, ttft_slo, tpot_slo)`，返回字典：

| 键 | 含义 |
| --- | --- |
| `ttft_p50`、`ttft_p99` | 首 token 延迟 TTFT = `first_token - arrival` 的分位数 |
| `tpot_p50`、`tpot_p99` | 每个输出 token 的时间 TPOT = `(finish - first_token) / (output_tokens - 1)`；只统计 `output_tokens >= 2` 的请求 |
| `e2e_p50` | 端到端延迟 `finish - arrival` 的中位数 |
| `throughput` | 输出 token 吞吐：所有 `output_tokens` 之和 / (最晚的 `finish` - 最早的 `arrival`) |
| `goodput` | 满足 SLO 的请求数 / 同一个时间跨度：满足 SLO 指 TTFT ≤ `ttft_slo`，并且（`output_tokens >= 2` 时）TPOT ≤ `tpot_slo` |
| `slo_attainment` | 满足 SLO 的请求比例 |

分位数用**线性插值**（`np.percentile` 的默认方法）。没有任何 `output_tokens >= 2` 的请求时，`tpot_p50`、`tpot_p99` 为 `0.0`。

<!-- 题解 -->
向量化：把字段取成 numpy 数组，`ttft = first - arrival`，`tpot` 用布尔掩码选出多 token 的请求再算。
goodput 是 DistServe 等论文提倡的指标：吞吐再高，如果大部分请求都违反了延迟约束，对用户也没有意义。
所以调度器优化的目标常常是"在满足 SLO 的前提下最大化吞吐"，而不是单纯的吞吐。
