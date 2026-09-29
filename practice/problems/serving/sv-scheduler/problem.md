---
title: 连续批处理调度器：token 预算、分块 prefill 与抢占
chapter: engine/scheduler.md
difficulty: 困难
tags: [调度, 连续批处理, 抢占]
---
实现正文 nano_engine 的 `Scheduler.schedule()`。每个请求只记录 `num_computed`（KV 已在缓存里的 token 数）和 `num_tokens`（提示词 + 已生成）；
每一步在 token 预算 `max_num_batched_tokens` 内给请求分配 token，让 `num_computed` 追上 `num_tokens`。模板里已经给出了 `Request`、`BlockPool`、`_allocate`、`_preempt`、`finish`。

`schedule()` 返回 `SchedulerOutput(scheduled, num_preempted)`，`scheduled` 是 `[(请求, 本步 token 数), ...]`。规则：

1. **先调度运行中的请求**，按 `self.running` 的顺序。对请求 `req`：`n = min(num_tokens - num_computed, 剩余预算)`；
   用 `_allocate(req, num_computed + n)` 保证块表够用。分配失败时，从 `running` **末尾**弹出一个请求抢占（`_preempt`），重试，直到分配成功；
   如果被抢占的正好是 `req` 自己，本轮不再调度运行队列里剩下的请求。预算用完（为 0）时停止；
2. **再从等待队列头部接收新请求**，条件：预算 > 0、**本步没有发生过抢占**、`len(running) < max_num_seqs`。对队首请求：
   `n = num_tokens - num_computed`；如果不允许分块 prefill（`enable_chunked_prefill=False`）且 `n > 预算`，停止接收；否则 `n = min(n, 预算)`；
   `_allocate` 失败也停止接收。成功时：出队、状态改为 `RUNNING`、追加到 `running` 末尾、加入 `scheduled`、扣预算。

（为了简化，这里去掉了前缀缓存相关的两行。）测试会用一个假的模型执行器反复调用 `schedule()`，比较每一步的调度结果。

<!-- 题解 -->
照正文的代码写，注意三个细节：

- `while not self._allocate(...)` 里抢占的是**末尾**的请求（最晚到达、优先级最低），被抢占的请求放回**等待队列最前面**，下次优先恢复；
- 抢占了自己之后要 `break` 出整个运行队列的循环；
- 抢占发生过（`num_preempted > 0`）的这一步不接收新请求：显存已经紧张，接收新请求只会导致下一步继续抢占，形成抖动。

vLLM V1 的调度器就是这个结构；SGLang 的调度更保守（按最坏情况预留，见 mini-sglang 的准入控制那道题），几乎不需要抢占。
