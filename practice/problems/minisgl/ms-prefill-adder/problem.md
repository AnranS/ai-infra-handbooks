---
title: PrefillAdder：准入控制与分块 prefill
chapter: schedule/scheduler.md
difficulty: 中等
tags: [调度, 准入控制, 分块 prefill]
---
实现 mini-sglang 组 prefill batch 的核心逻辑（纯 Python 简化版）。等待中的请求 `PendingReq` 有 `uid`、`input_len`、`output_len`、`cached_len`（前缀缓存命中的长度，由测试给出）、`chunked`（上一轮是否没做完分块，以及已经做到的位置）。

`schedule_prefill(pending, token_budget, reserved_size, available_kv, free_rows)` 按顺序（先来先服务）处理 `pending`，返回 `(batch, new_pending)`：

1. token 预算用完（`<= 0`）就停；
2. 如果请求处于分块中（`req.chunked is not None`，值是已经算完的长度 `done`）：资源早已分配好，直接从 `done` 接着做：本轮算 `min(预算, input_len - done)` 个；
3. 否则做**准入控制**：需要一个空闲行（`free_rows > 0`），并且最坏情况下的 KV 需求 `input_len - cached_len + output_len` 加上 `reserved_size` 不超过 `available_kv`；不满足就**停止**（后面的请求也不看）。准入成功时占用一行（`free_rows -= 1`），本轮从 `cached_len` 开始算 `min(预算, input_len - cached_len)` 个；
4. 每接收一个请求：预算减去本轮算的 token 数；**只有新准入的请求**才让 `reserved_size += input_len - cached_len + output_len`（分块中的请求准入时已经预留过）；
5. 本轮没算完的请求是"分块中"：`chunked` 记为新的已完成长度；
6. `batch` 是 `[(uid, start, end), ...]`（本轮算 `[start, end)` 这些位置）；`new_pending`：**本轮没做完的分块请求排在最前面**，然后是没被处理的请求，已经做完 prefill 的请求离开队列。

返回的 `new_pending` 里的对象可以是原来的对象（修改它们的 `chunked` 字段）。

<!-- 题解 -->
对应书中 `PrefillAdder.try_add_one` 和 `PrefillManager.schedule_next_batch`：

```python
for req in pending:
    if budget <= 0: break
    if req.chunked is not None: start = req.chunked
    else:
        need = req.input_len - req.cached_len + req.output_len
        if free_rows == 0 or need + reserved > available_kv: break
        free_rows -= 1; reserved += need; start = req.cached_len
    n = min(budget, req.input_len - start); budget -= n
    ...
```

为什么预留"最坏情况"：这样运行中的请求永远不会因为 KV 不够而被抢占（第 8 章）。分块中的请求排到队首，下一轮优先继续，避免它长期占着资源却迟迟不完成。
