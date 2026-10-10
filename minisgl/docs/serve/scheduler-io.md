# 调度器的收发与多 rank 同步

<p class="lead">调度器的主循环只调用两个 IO 方法：<code>receive_msg(blocking)</code> 和 <code>send_result(reply)</code>。它们背后是 <code>SchedulerIOMixin</code>：单卡时直接收发 ZMQ；张量并行时只有 rank 0 连接 tokenizer，并且要保证所有 rank 在每一轮看到完全相同的消息。这一章实现它，并解释为什么"完全相同"如此重要。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 张量并行时，为什么每个 rank 都要运行一个完整的调度器，而不是只让 rank 0 调度、其他 rank 听指挥？
    2. rank 0 每轮收到的消息条数不固定。其他 rank 怎么知道这一轮该收几条？
    3. 谁负责把结果发给 detokenizer？
    4. `receive_msg(blocking=True)` 什么时候被调用？阻塞之前做了什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 每个 rank 的 KV 池、页表都在自己的 GPU 上，调度决定直接决定 KV 的布局；让每个 rank 都跑完整的调度器，只要输入消息相同，确定性的调度就会得到相同的 batch 和 KV 布局，不需要每一轮再把调度结果（很多元数据）发给其他 rank。
    2. rank 0 用一次 broadcast 告诉其他 rank 本轮的消息条数，其他 rank 按这个条数从 PUB / SUB 里收取。
    3. 只有 rank 0：各 rank 采样出的 token 本来就相同，只需要一份。
    4. 当前没有任何工作（没有运行中的请求、没有待处理的请求）时才阻塞等待新消息，免得空转；阻塞之前要先处理完所有在途的结果、做空闲时的检查。

**本章要写的文件**：`scheduler/io.py`。

@@tree@@

这一步的 main：`examples/ch14_scheduler_io.py`——它只用到上面这些文件；`python tools/steps.py check` 会逐章搭出这棵树、跑这个 main。

## 为什么每个 rank 都调度

张量并行时，每个 rank 持有模型的一部分和 KV 缓存的一部分（按 KV 头切分，第 16 章）。每一轮，所有 rank 必须计算**同一个 batch**：同样的请求、同样的顺序、同样的 token 写到 KV 池的同样位置。否则 all-reduce 会把不同请求的部分和加在一起。

一种做法是只让 rank 0 调度，把每轮的 batch（请求列表、位置、页分配……）广播给其他 rank。mini-sglang 的做法更省通信：**每个 rank 都运行一个完整的调度器，只要它们看到的输入消息完全相同，确定性的调度逻辑就会在每个 rank 上做出完全相同的决策**，KV 池、page table、Radix Cache 在各 rank 上始终保持一致。需要广播的只有原始消息，每条几百字节。

这要求调度逻辑是确定性的——第 7 章 `DecodeManager` 组 batch 时按 `uid` 排序，就是为了消除 `set` 遍历顺序在不同进程中的差异。

## 收消息

@@code python/minisgl/scheduler/io.py:SchedulerIOMixin.__init__@@

构造时按"是不是 rank 0、有没有其他 rank"选好收发方法。离线模式直接换成 `LLM` 提供的两个方法（第 7 章）。

单卡的版本：

@@code python/minisgl/scheduler/io.py:SchedulerIOMixin._recv_msg_single_rank@@

没事可做时（`blocking=True`）先调用 `run_when_idle`（做一次内存完整性检查，第 8 章），再阻塞等第一条消息；然后把已经到达的消息一次取完，不再等待。

多卡时 rank 0：

@@code python/minisgl/scheduler/io.py:SchedulerIOMixin._recv_msg_multi_rank0@@

1. 阻塞时收到的第一条消息，立即原样转发；
2. 把已经到达的消息取完（这一步取到多少条，只有 rank 0 知道）；
3. **用一次 broadcast 告诉其他 rank 这一轮还有几条**；
4. 原样转发这些消息，自己解码。

其他 rank：

@@code python/minisgl/scheduler/io.py:SchedulerIOMixin._recv_msg_multi_rank1@@

阻塞时先从 PUB/SUB 收一条（与 rank 0 的第一条对应），再参与 broadcast 得到条数，按条数收。两边的 `blocking` 参数总是相同的——它取决于调度器状态，而各 rank 的状态完全一致。

广播条数用的是 CPU 上的 gloo 进程组（`tp_cpu_group`）。GPU 上时，张量走 NCCL，控制信息走另一个 gloo 组，互不干扰。

## 看两个 rank 收到了什么

不加载模型，只用调度器的 IO 部分：主进程扮演 tokenizer 连续发 3 条消息，两个进程扮演两个 rank。

@@code examples/ch14_scheduler_io.py@@

@@output ch14_scheduler_io@@

rank 0 第一轮阻塞收到第 1 条，再取走当时已经到达的若干条；rank 1 通过 broadcast 得知条数，从 PUB/SUB 收到同样的消息。无论消息怎样分批到达，两个 rank 收到的序列都相同。（这个演示第一次运行时卡死了，原因是 PUB/SUB 的"慢订阅者"问题，见上一章的"与官方的差异：XPUB"。）

## 发结果

@@code python/minisgl/scheduler/io.py:SchedulerIOMixin._reply_tokenizer_rank0@@

只有 rank 0 把结果发给 detokenizer，其他 rank 的 `send_result` 什么也不做。所有 rank 的采样结果本来就相同：LM head 的 logits 经过 all-gather 后每个 rank 都有完整的一份，贪心解码自然相同；随机采样时各 rank 用同一个随机种子（`Engine` 里的 `torch.manual_seed(42)`）和同样的调用序列，结果也相同。

!!! upstream "官方实现"
    - @@upstream scheduler/io.py:SchedulerIOMixin@@
    - 官方最新的一次提交（#113）修的就是"各 rank 的 decode batch 请求顺序不一致"——一个直接违反"完全相同"原则的问题。

## 测试

多 rank 的收发由第 16 章的张量并行端到端测试覆盖：TP=2 和 TP=4 的服务，输出与单卡的 Hugging Face 完全一致。

!!! interview "怎么讲清楚"
    讲多 rank 同步：张量并行时每个 rank 都运行一个完整的调度器，只要所有 rank 收到完全相同的消息，确定性的调度就会组出相同的 batch、分配相同的 KV 位置，不需要 rank 0 每轮把调度结果发下去。做法是 rank 0 收消息、原样转发原始字节，并用一次 broadcast 告诉其他 rank 本轮有几条，其他 rank 按条数从 PUB/SUB 收取；只有 rank 0 把结果发给 detokenizer（各 rank 的采样结果本来就相同）。没有任何请求时才阻塞等待消息，阻塞前先把在途的 batch 处理完。确定性的前提：调度不依赖各 rank 不同的状态（比如时间、集合的迭代顺序，所以要按 uid 排序）。

## 练习

1. 随机采样时，如果各 rank 的随机数生成器状态不同，会发生什么？只有 rank 0 发结果，看起来没问题，实际呢？
2. 如果把"广播条数"去掉，让 rank 1 用 `empty()` 自己判断还有没有消息，会有什么问题？
3. 为什么 rank 0 要"先转发、再解码"，而不是"先解码、再编码转发"？

??? success "参考答案"
    1. 各 rank 会采样出不同的 token，而 token pool 由各 rank 自己写入：下一轮各 rank 的输入不同，all-reduce 把不同序列的数据加在一起，所有 rank 的计算都错了（只是 rank 0 的输出看起来仍是"某个"序列）。所以各 rank 必须用相同的随机种子，并保证调用随机数的次数和顺序相同。
    2. PUB/SUB 的消息到达 rank 1 有延迟，rank 1 判断"没有消息了"时，rank 0 可能已经转发了几条还在路上。两边这一轮处理的消息集合不同，调度决策就会分叉。
    3. 省一次编码，也保证转发的是逐字节相同的内容。

## 小结

- [x] 每个 rank 都运行完整的调度器，只要输入消息相同，确定性的调度就会得到相同的 batch 和相同的 KV 布局。
- [x] rank 0 原样转发消息，并用一次 broadcast 告诉其他 rank 本轮条数；其他 rank 按条数从 PUB/SUB 收取。
- [x] 只有 rank 0 回复 detokenizer；各 rank 的采样结果本来就相同。
