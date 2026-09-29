# 调度器骨架与离线接口

<p class="lead">上一章我们手工扮演了调度器：分配 KV 位置、填 batch 字段、追加 token。这一章把它自动化。mini-sglang 的调度器由四个小管理器组成——请求表、KV 缓存、decode 集合、prefill 队列——主循环只有十几行：收消息、选一个 batch、准备、前向、处理结果。我们先实现最朴素的版本（不复用前缀、不分块、不重叠），并用离线接口 <code>LLM</code> 驱动它。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 连续批处理（continuous batching）和静态批处理的区别是什么？
    2. mini-sglang 一轮只做 prefill 或只做 decode，还是会混在一起？优先做哪个？
    3. decode 集合是一个 `set`，组 batch 时为什么要按 `uid` 排序？
    4. 离线接口 `LLM` 没有 ZMQ，它怎样复用调度器的主循环？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 静态批处理要等一整批请求全部结束才换下一批；连续批处理每一轮都重新组 batch，结束的请求立刻离开、新请求随时加入。
    2. 一轮只做一种：有能接纳的 prefill 就优先做 prefill，否则做 decode。
    3. 张量并行时每个 rank 都运行自己的调度器，必须组出完全相同的 batch（请求顺序决定了 KV 的布局）；`set` 的遍历顺序不确定，按 `uid` 排序保证各 rank 一致。
    4. `LLM` 继承调度器，覆盖"接收消息"和"发送结果"两个方法：直接从内存里的请求列表取输入、把结果收集到本地，主循环完全复用。

**本章要写的文件**：`scheduler/config.py`、`scheduler/utils.py`、`scheduler/decode.py`、`scheduler/prefill.py`、`scheduler/scheduler.py`、`llm/llm.py`（`scheduler/table.py` 上一章已写；`scheduler/cache.py` 下一章写，本章用它的 naive 模式）。

@@video batching 动画：连续批处理与准入控制（约 1.5 分钟，覆盖本章和下一章）@@

## 组成

@@code python/minisgl/scheduler/scheduler.py:Scheduler.__init__@@

@@diagram scheduler-loop 调度器的组成与主循环@@

| 管理器 | 职责 | 章节 |
| --- | --- | --- |
| `TableManager` | page table / token pool 的行：每个运行中的请求一行 | [KV 池](../compute/kvcache.md) |
| `CacheManager` | KV 池的页分配与回收，前缀缓存的门面 | [CacheManager](cache-manager.md) |
| `DecodeManager` | 正在 decode 的请求集合 | 本章 |
| `PrefillManager` | 等待 prefill 的请求队列，组 prefill batch | 本章，分块见[第 10 章](chunked-prefill.md) |

调度器还继承了 `SchedulerIOMixin`，提供 `receive_msg` 和 `send_result`，在线服务时它们走 ZMQ（第 14 章），离线时由 `LLM` 覆盖。

## 主循环

@@code python/minisgl/scheduler/scheduler.py:Scheduler.normal_loop@@

每一轮四步：

1. **收消息**。如果手上没有任何可以做的事（没有等待 prefill 的，也没有正在 decode 的），就阻塞等待；否则只取走已经到达的消息，不等待。
2. **选 batch**。有等待的请求就做 prefill，否则做 decode：

    @@code python/minisgl/scheduler/scheduler.py:Scheduler._schedule_next_batch@@

    这是"prefill 优先"策略：新请求尽快得到第一个 token（TTFT 低），代价是正在 decode 的请求会被 prefill 打断一轮。一轮只做一种，不混合——这让每轮的 batch 形状简单（decode batch 可以用 CUDA Graph），也是 mini-sglang 比 SGLang 正式版简单的地方之一。
3. **准备并前向**。`_prepare_batch` 把 batch 的各个字段填好，`_forward` 调用引擎。
4. **处理结果**。把新 token 交给请求，判断结束，回复 detokenizer，释放结束请求的资源。

`run_forever` 就是反复调用这个循环（重叠调度的版本 `overlap_loop` 在第 11 章）。

## 两个管理器

`PrefillManager` 维护一个先来先服务的等待队列，每轮用一个 `PrefillAdder` 从队首开始尽量多地接纳请求：

@@code python/minisgl/scheduler/prefill.py:PrefillManager.schedule_next_batch@@

`PrefillAdder` 为每个请求做三件事：在前缀缓存里匹配已缓存的部分、检查请求槽和 KV 空间够不够（准入控制，下一章详讲）、在 token pool 里写入本轮要算的 token。遇到第一个放不下的请求就停止——不跳过它去接纳后面更小的请求，这样保证公平，不会有请求一直被插队。

`DecodeManager` 更简单，就是一个请求集合：

@@code python/minisgl/scheduler/decode.py:DecodeManager@@

每轮前向之后，`filter_reqs` 把本轮的请求并进来，并去掉已经不能再生成的（`can_decode` 为假）。刚做完 prefill 的请求就这样自然地加入了 decode 集合。组 batch 时按 `uid` 排序：张量并行下每个 rank 都有一个 `DecodeManager`，Python 的 `set` 遍历顺序取决于对象的内存地址，不同进程里可能不同；而各 rank 的 batch 必须完全一致（同样的请求、同样的顺序），否则 all-reduce 就把不同请求的数据加在了一起。官方仓库最新的一次提交修的正是这个问题。

## 准备 batch

@@code python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch@@

按顺序：

1. **补齐**（第 18 章，不用 CUDA Graph 时什么也不做）；
2. **分配 KV 页**：为本轮要算的 token 分配位置并写进 page table（第 8 章）；
3. **位置**：每个请求 `[cached_len, device_len)`，拼成一维；
4. **两组二维下标**：`input_tuple = (行号, 位置)` 用来从 token pool 取输入、从 page table 取 `out_loc`；`write_tuple = (行号, device_len)` 用来把采样结果写回 token pool 的下一个位置；
5. **注意力元数据**和**采样参数**。

@@code python/minisgl/scheduler/scheduler.py:_make_write_tuple@@

然后前向：

@@code python/minisgl/scheduler/scheduler.py:Scheduler._forward@@

输入 `token_pool[input_tuple]` 和写回 `token_pool[write_tuple] = next_tokens` 都是设备上的索引操作，CPU 不需要知道 token 的值。

## 处理结果

@@code python/minisgl/scheduler/scheduler.py:Scheduler._process_last_data@@

对 batch 中的每个请求：把新 token 追加到 CPU 上的 `input_ids`，判断是否结束（达到 `max_tokens`，或者遇到 EOS 且没有设置 `ignore_eos`），生成一条 `DetokenizeMsg`。结束的请求释放资源：归还 page table 的行，把 KV 交给缓存（naive 缓存会全部释放）。刚做完 prefill 的请求，把提示词的 KV 交给前缀缓存（第 9 章）。

这段代码里有两处是为第 11 章的重叠调度准备的——`finished_reqs` 和"用 `len(req.input_ids)` 而不是 `req.can_decode` 判断长度"——那一章再解释。

新请求的处理：

@@code python/minisgl/scheduler/scheduler.py:Scheduler._process_one_msg@@

提示词加上 `max_tokens` 超过模型最大长度时，`max_tokens` 被截短；提示词本身就超长的请求直接丢弃（官方同样只打一条警告，前端不会收到任何回复——这是一个可以改进的地方）。

## 离线接口 LLM

@@code python/minisgl/llm/llm.py:LLM@@

`LLM` 继承 `Scheduler`，用 `offline_mode=True` 让 IO 层不创建任何 ZMQ 队列，然后覆盖两个方法：

- `offline_receive_msg`：从待处理列表里取请求，按 prefill 预算（`max_extend_tokens`）分批放入。当调度器"没事可做、要阻塞等待新消息"而列表也空了时，说明全部完成，抛出 `RequestAllFinished` 跳出 `run_forever`；
- `offline_send_result`：把每条回复的 token 记到对应请求上（EOS 不记）。

这个小技巧让离线接口与在线服务共用同一个主循环，测过一个就等于测过另一个的调度逻辑。

## 运行

@@code examples/ch07_scheduler.py@@

@@output ch07_scheduler@@

第一轮 5 个请求一起 prefill（24 个 token 拼成一维）。`max_tokens=1` 的 uid3 在 prefill 之后就结束了，没有进入 decode；uid1 在一轮 decode 后离开；其余请求继续，batch 逐渐变小。这就是连续批处理：请求按自己的节奏加入和离开，不用等整个 batch 结束。5 个请求的输出都与 Hugging Face 单独生成的结果相同。

!!! upstream "官方实现"
    - 主循环：@@upstream scheduler/scheduler.py:Scheduler.normal_loop@@
    - 准备与前向：@@upstream scheduler/scheduler.py:Scheduler._prepare_batch@@、@@upstream scheduler/scheduler.py:Scheduler._forward@@
    - prefill 队列：@@upstream scheduler/prefill.py:PrefillManager.schedule_next_batch@@
    - decode 集合：@@upstream scheduler/decode.py:DecodeManager@@
    - 离线接口：@@upstream llm/llm.py:LLM@@

## 测试

@@code tests/test_ch07_scheduler.py:test_requests_with_different_lengths_leave_and_join@@

## 练习

1. 把调度策略改成"decode 优先"：有正在 decode 的请求就先做 decode，没有时才做 prefill。会有什么问题？（提示：考虑一个长期有请求在 decode 的服务里，新请求什么时候能被处理。）
2. `PrefillManager` 遇到第一个放不下的请求就停止。如果改成"跳过它、继续尝试后面的请求"，吞吐会怎样？有什么风险？
3. 在 `_process_one_msg` 里，超长的提示词被直接丢弃，前端永远收不到回复。怎样改成给前端返回一个错误？需要改哪些消息类型？

??? success "参考答案"
    1. 只要 decode 集合不空，新请求就永远得不到 prefill，TTFT 无上限（饥饿）。实用的折中是限制 decode 连续执行的轮数，或者像 SGLang 正式版那样把 prefill 和 decode 混在一个 batch 里。
    2. 吞吐可能提高（小请求填满了预算），但大请求可能一直被后来的小请求插队，出现饥饿。需要"老化"机制：等待越久优先级越高。
    3. 调度器直接回复一条带错误标记的 `DetokenizeMsg`（例如新增 `error` 字段，`finished=True`），detokenizer 转成带错误信息的 `UserReply`，API Server 据此返回 400。

## 小结

- [x] 调度器 = 四个管理器 + 一个主循环：收消息 → 选 batch（prefill 优先，一轮只做一种）→ 准备 → 前向 → 处理结果。
- [x] prefill 队列先来先服务、遇到放不下的就停；decode 集合在每轮前向后更新，组 batch 时按 uid 排序保证各 rank 一致。
- [x] 输入从 token pool 取、采样结果写回 token pool，都在设备上完成。
- [x] 离线接口 `LLM` 覆盖收发两个方法，复用同一个主循环。
