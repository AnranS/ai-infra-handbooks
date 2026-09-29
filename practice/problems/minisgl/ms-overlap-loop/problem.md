---
title: 重叠调度的结果处理
chapter: schedule/overlap.md
difficulty: 困难
tags: [重叠调度, 状态机, 修 bug]
---
重叠调度"先发射第 N+1 轮、再处理第 N 轮"，CPU 上的状态比 GPU 的结果提前一轮。这道题用一个简化的模型复现书中发现的前两个问题，并修好它们。

模型：每个请求 `Req(uid, prompt_len, max_tokens)`；`device_len` 初始为 `prompt_len`，`max_device_len = prompt_len + max_tokens`，`input_len`（CPU 上已知的 token 数）初始为 `prompt_len`。
GPU 上第 `n` 次为某个请求生成的 token 由 `sample(uid, n)` 决定（`n` 从 0 开始，每发射一次加 1）。

- **发射**一轮：对所有"运行中"的请求调用 `complete_one`（`device_len += 1`），并在 GPU 上生成 token（记下 `sample(uid, n)`）；
- **处理**一轮的结果：对这一轮里的每个请求，把 token 追加到 CPU（`input_len += 1`），判断是否结束，发出消息 `(uid, token, finished)`；结束的请求从运行队列移除。

结束条件：token 是 `eos`，或者**已经收到的 token 数达到上限**（`input_len >= max_device_len`）。

实现 `overlap_loop(reqs, sample, eos)`：所有请求一开始都在运行队列里；循环"发射本轮（运行队列里的全部请求）→ 处理上一轮"，直到没有运行中的请求、也没有待处理的结果。返回消息列表（按处理顺序）。要求：

1. 每个请求收到的消息序列与普通循环（发射后立刻处理）完全相同：结束标记恰好出现在最后一条消息上；
2. 一个请求在第 N 轮遇到 EOS 时，它已经被发射进了第 N+1 轮（这是重叠调度的代价，允许），但处理第 N+1 轮时**不能再为它发任何消息**；
3. 模板是按官方写法实现的（`finished = not can_decode`，处理结果时不跳过已结束的请求），会同时触发这两个问题。

<!-- 题解 -->
问题一：处理第 N 轮时第 N+1 轮已经发射，`device_len` 已经推进了一步，`can_decode`（`max_device_len - device_len > 0`）反映的是"第 N+1 轮之后"的状态，
于是结束标记提前了一个 token。改用只取决于已收到 token 数的条件：`input_len >= max_device_len`。

问题二：维护一个集合 `finished_prev`：处理第 N 轮时结束的请求记进去；处理第 N+1 轮时，跳过这个集合里的请求（它们的结果是过期的），然后清空、换成本轮新结束的请求。
