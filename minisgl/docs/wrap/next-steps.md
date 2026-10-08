# 与 SGLang 的差距和扩展练习

<p class="lead">到这里，你手上有一个完整的 mini-sglang：OpenAI 兼容服务、Radix Cache、分块 prefill、重叠调度、张量并行、三种注意力后端、CUDA Graph、MoE。它与官方同结构、同接口，并且修正了几个官方的问题。这一章盘点它与 SGLang 正式版的差距，每一项都是一个可以继续做下去的方向，并附上在代码里从哪里下手。</p>

## 回顾：一个请求经过的代码

| 步骤 | 代码 | 章节 |
| --- | --- | --- |
| HTTP 请求进来，分配 uid | `server/api_server.py` `v1_chat_completions` | 15 |
| 套模板、分词 | `tokenizer/tokenize.py` | 13 |
| 进入等待队列 | `scheduler/scheduler.py` `_process_one_msg` | 7 |
| 匹配前缀、准入、分配请求槽 | `scheduler/prefill.py` `PrefillAdder` | 8、9 |
| 分块 | `PrefillAdder._add_one_req` | 10 |
| 补齐、分页、位置、元数据 | `Scheduler._prepare_batch` | 7、8、18 |
| 前向（或 replay）、采样 | `Engine.forward_batch` | 6、18 |
| 注意力 | `attention/*.py` | 5、17 |
| 张量并行通信 | `layers/linear.py`、`distributed/` | 16 |
| 处理结果、缓存前缀、释放 | `Scheduler._process_last_data`、`CacheManager.cache_req` | 7、9、11 |
| 增量反分词 | `tokenizer/detokenize.py` | 13 |
| SSE 流式返回 | `FrontendManager.stream_chat_completions` | 15 |

## 本书修正的官方问题

复刻过程中发现并修正了下面五处。每处都有测试：把修正改回官方写法，测试就会失败。它们可以整理成给官方仓库的 issue 或 PR：

| 问题 | 影响 | 章节 |
| --- | --- | --- |
| 重叠调度下用 `can_decode` 判断达到 `max_tokens` | finished 标记提前一个 token，在线服务少返回一个 token | [重叠调度](../schedule/overlap.md) |
| EOS 之后被多调度的一轮仍然发送回复 | detokenizer 为已结束的请求新建状态，永不释放 | [重叠调度](../schedule/overlap.md) |
| 结束请求的请求槽立即复用 | 调度器 stream 与引擎 stream 对同一行 token pool 的写入竞争 | [重叠调度](../schedule/overlap.md) |
| prefill 在途时收到 abort | 对已释放的请求再次 `cache_req`，Radix Cache 下页被重复释放 | [重叠调度](../schedule/overlap.md) |
| PUB/SUB 没有等待订阅者 | 第一条广播可能丢失，其他 rank 永久阻塞（依赖启动时序才不出现） | [消息与 ZMQ](../serve/message.md) |

另外两处值得一提：官方 Radix Cache 的完整性检查是空的，我们实现了真正遍历树的检查（第 9 章）；官方的 YaRN 没有乘 Hugging Face 实现里的注意力缩放系数，我们没有实现 YaRN，把它留作练习（第 2 章）。

## 与 SGLang 正式版的差距

**调度**

- **混合 batch**：mini-sglang 一轮只做 prefill 或只做 decode，长提示词分块期间 decode 暂停。正式版把 decode 请求和 prefill 块放进同一个 batch。从 `Scheduler._schedule_next_batch` 下手（第 10 章练习 1）。
- **抢占与重算**：mini-sglang 按最坏情况预留 KV，从不抢占，但并发偏低。正式版乐观地接纳，KV 不够时撤回（retract）部分请求，靠 Radix Cache 让重算很便宜。从 `PrefillAdder._try_allocate_one` 与 `_prepare_batch` 下手（第 8 章练习 2）。
- **缓存感知的调度策略**：正式版有 `lpm`（最长前缀匹配优先）、`dfs-weight` 等策略，优先调度能命中更多缓存的请求；mini-sglang 是先来先服务。从 `PrefillManager.schedule_next_batch` 的遍历顺序下手。

**KV 缓存**

- **分层缓存（HiCache）**：把被淘汰的 KV 先放到 CPU 内存甚至磁盘，而不是直接丢弃。官方 `MatchResult` 里留了一个 "TODO: support HiCache"。
- **MLA 与其他注意力变体**：`create_kvcache_pool` 只支持 MHA/GQA（官方注释 "TODO: support other variants (e.g. MLA)"）。支持 DeepSeek 需要新的 KV 池布局和注意力后端。
- **KV 缓存量化**：FP8 KV 缓存能把容量翻倍。

**分布式**

- **PD 分离**：prefill 和 decode 放在不同的实例上，中间传输 KV（原理见[推理系统手册](serving://distributed/pd-disagg/)）。
- **专家并行与 DP Attention**：MoE 模型的主流部署方式，mini-sglang 只有张量并行。
- **路由**：多个实例前面的缓存感知路由器（SGLang 的 sgl-router）。

**解码功能**

- **投机解码**：EAGLE、MTP 等，需要草稿模型、树形验证、KV 回滚。
- **结构化输出**：用语法约束采样（xgrammar），需要在采样前对 logits 加掩码。
- **更多采样参数**：停止字符串、重复惩罚、logprobs、`n > 1` 的并行采样、min-p。当前的 `SamplingParams` 只有温度、top-k、top-p、`ignore_eos`、`max_tokens`。

**其他**

- 量化模型（FP8、AWQ、GPTQ）、LoRA、多模态、可观测性（Prometheus 指标）、健康检查与优雅退出。

## 扩展练习

按难度排序，每个都可以作为一个作品集项目（写法见[推理系统手册的作品集一章](serving://career/projects/)）：

1. **补全采样参数**：实现停止字符串、`min_p`、`logprobs`。改动 `SamplingParams`、`Sampler`、`DetokenizeMsg` 和 API Server。
2. **给官方提 PR**：挑本书修正的一个问题，在官方仓库写一个最小复现（GPU 上），提交修复。先读官方的贡献指南，PR 描述里说明复现方法和测试结果。
3. **混合 batch**：让 prefill 块与 decode 请求共享一个 batch，用第 21 章的压测对比长提示词下 TPOT 的 P99。
4. **抢占**：实现乐观准入 + 抢占，比较相同 KV 容量下的并发数与吞吐。
5. **HiCache**：把淘汰的 KV 拷到 CPU 内存，命中时拷回。需要在 `RadixTreeNode` 上区分 GPU 与 CPU 上的 value。
6. **PD 分离原型**：两个进程，一个只做 prefill、一个只做 decode，用共享内存（CPU 上）或 NCCL（GPU 上）传 KV。

## 面试怎么讲

!!! interview "怎么介绍你实现的推理引擎"
    按"一个请求的旅程"讲：进程结构 → 调度器的四个管理器 → 连续批处理与准入控制 → Radix Cache → 重叠调度 → 张量并行 → CUDA Graph。每一部分给一个数字（例如"共享 400 token 前缀时 prefill 计算量降到十分之一""TP=2 时每次前向 57 次 all-reduce"），并准备一个深入的细节：重叠调度下的那几个问题最能体现你真的把代码吃透了——为什么状态会领先一轮、会导致什么、怎样修、怎样测。

## 小结

- [x] 一个请求经过的每一步都对应本书的一章，也对应官方仓库的一个文件。
- [x] 本书修正了官方的五处问题，每处都有能复现的测试。
- [x] 与正式版的差距集中在调度策略（混合 batch、抢占、缓存感知）、KV 缓存（分层、MLA、量化）、分布式（PD 分离、EP）和解码功能（投机解码、结构化输出）。
