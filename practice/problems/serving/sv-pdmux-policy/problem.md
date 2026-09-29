---
title: PD 复用：按架构切分 SM，按 decode 负载选档
chapter: frontier/pd-multiplex.md
difficulty: 中等
tags: [PD 复用, green context, SGLang, 调度]
---
SGLang 的 PD 复用在启动时把 GPU 的 SM 切成几档 (prefill SM, decode SM)，运行时按正在 decode 的请求数选一档。按 `srt/multiplex/` 的规则实现：

1. `divide_sm(total_sms, major, groups)`：计算能力大版本 `major` 决定 (每份最少 SM 数, SM 数的粒度)：6 → (1, 1)，7 → (2, 2)，8 → (4, 2)，9 → (8, 8)，其他抛出 `ValueError`。候选的 prefill SM 数 x 从"最少 SM 数"开始、按粒度递增，不超过 `total_sms - 最少 SM 数`，并且满足 prefill 不少于一半（`x >= total_sms - x`）、decode 至少 16 个（`total_sms - x >= 16`）。没有候选时抛出 `ValueError`；候选数不少于 `groups` 时，以 `step = max(1, 候选数 // groups)` 为步长隔一个取一个、取前 `groups` 个；最后返回 `[(x, total_sms - x), ...]`，prefill 多的排在前面；
2. `stream_groups(total_sms, major, sm_group_num=8)`：第 0 组整卡给 prefill `(total_sms, 0)`，中间是 `divide_sm(..., sm_group_num - 2)` 的结果，最后一组整卡给 decode `(0, total_sms)`；
3. `choose(groups, decode_bs, has_prefill, decode_bs_divisor=36, thresholds=None)`：返回选中的组号。同时有 decode 请求和 prefill 时：没有 `thresholds` 就用 `max(1, min(n - 2, decode_bs × (n - 2) // decode_bs_divisor))`（n 是组数）；有 `thresholds`（中间每一组一个阈值，从第 1 组开始）时，选"阈值不超过 decode_bs 的最后一组"，一个都不满足时选第 1 组。只有 decode 时选最后一组，没有 decode 请求时选第 0 组；
4. `prefill_layers_per_step(extend_tokens, num_layers, done_layers=0, token_budget=65536)`：prefill 按层切开，这一轮算多少层：`max(1, token_budget // extend_tokens)`，但不超过剩下的层数；`extend_tokens` 为 0 时一次算完剩下的层。

```python
stream_groups(132, 9)       # H100：[(132, 0), (112, 20), (104, 28), (96, 36), (88, 44), (80, 52), (72, 60), (0, 132)]
choose(stream_groups(132, 9), 32, True)    # 5：32 个 decode 请求时选 (80, 52)
```

<!-- 题解 -->
粒度来自 green context 的限制：CUDA 规定计算能力 9.0 及以上按 8 个 SM 切分，7.x、8.x 按 2 个。"prefill 不少于一半、decode 至少 16 个"是 SGLang 的取舍：decode 受访存限制，少量 SM 就能接近带宽上限，把大部分算力留给受算力限制的 prefill。

选档是一个线性映射：decode 请求越多，需要的带宽和算力越多，档位越往 decode 多的方向走；默认的 `decode_bs_divisor = 36` 意味着 36 个请求以上就用 decode 最多的一档。只有一边有活时用整卡的普通流。prefill 按层切开，每轮只算 `65536 // token 数` 层，保证一轮的时间有上限、decode 不会被一个大 prefill 拖住，prefill 做完再换档（换档要同步两个流）。
