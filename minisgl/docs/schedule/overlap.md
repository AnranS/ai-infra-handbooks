# 重叠调度：藏起 CPU 开销

<p class="lead">一个小模型 decode 一步，GPU 只要几毫秒；而调度器每一轮在 CPU 上要做的事——收消息、组 batch、准备元数据、处理上一轮的结果、发消息——同样要几毫秒。如果两者轮流进行，GPU 有将近一半的时间在等 CPU。重叠调度（出自 NanoFlow，SGLang 0.4 引入）让 CPU 在 GPU 计算第 N+1 轮的同时处理第 N 轮的结果。mini-sglang 的实现只多了十几行，但它改变了"什么时候知道什么"，由此引出了本书在复刻时发现的四个问题。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 发射第 N+1 轮时，CPU 还不知道第 N 轮采样出的 token。第 N+1 轮的输入从哪来？
    2. 一个请求在第 N 轮采样出了 EOS，它会不会出现在第 N+1 轮？
    3. 为什么调度器和引擎要用两条不同的 CUDA stream？
    4. 重叠调度下，处理第 N 轮结果时，`req.device_len` 反映的是第几轮之后的状态？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 从 GPU 上的 token pool：第 N 轮在 GPU 上采样出的 token 直接写进 token pool，第 N+1 轮的前向从 token pool 读取输入，CPU 不需要知道它是什么。
    2. 可能会：发射第 N+1 轮时 CPU 还没处理第 N 轮的结果，不知道它已经结束，所以多调度了一轮；处理结果时丢弃这个多余的 token。
    3. 让调度器准备下一轮（拷贝元数据、组 batch）的工作和引擎的计算在不同的 stream 上并行，并用事件建立依赖；都放在一条 stream 上就又串行了。
    4. 第 N+1 轮之后：CPU 上的请求状态在发射时就提前推进了，比正在处理的结果领先一轮。

**本章要写的文件**：`scheduler/scheduler.py` 中的 `overlap_loop`、`run_forever`，以及 `_process_last_data`、`_free_req_resources`、`_process_one_msg` 里与重叠有关的部分。

@@video overlap 动画：重叠调度，以及它带来的四个问题（约 2 分钟）@@

## 两种循环

@@code python/minisgl/scheduler/scheduler.py:Scheduler.overlap_loop@@

与 `normal_loop` 比，只有一处不同：**先发射本轮，再处理上一轮**。`overlap_loop` 接收上一轮的数据、返回本轮的数据，`run_forever` 把它们串起来：

@@code python/minisgl/scheduler/scheduler.py:Scheduler.run_forever@@

用一个请求、3 个 token 看事件顺序（示例的前半部分）：

```text
普通循环：  发射1 处理1 发射2 处理2 发射3 处理3
重叠循环：  发射1 发射2 处理1 发射3 处理2 处理3
```

在 GPU 上，"发射"只是把 kernel 放进 stream 的队列，立即返回；"处理"要等结果拷回 CPU。重叠循环里，处理第 1 轮时 GPU 已经在算第 2 轮，CPU 的开销被藏在 GPU 计算的时间里：

@@diagram overlap-timeline 普通循环与重叠循环的 CPU / GPU 时间线@@

## 三个前提

能这样做，依赖前面几章早已埋下的三个设计。

**第一，下一轮的输入不需要经过 CPU。** 第 N+1 轮 decode 的输入是第 N 轮采样出的 token，发射第 N+1 轮时 CPU 还不知道它是什么。但 `_forward` 把采样结果直接在 GPU 上写进了 token pool 的下一个位置，第 N+1 轮再从 token pool 里取输入（第 7 章的 `write_tuple` 和 `input_tuple`）。GPU 按 stream 顺序执行，读一定发生在写之后。

**第二，调度所需的状态在 CPU 上提前推进。** `Engine.forward_batch` 在发射后立即对每个请求调用 `complete_one()`，所以调度第 N+1 轮时，请求的 `cached_len`、`device_len` 已经是"第 N 轮之后"的值，可以正确地算出位置、分配 KV 页。唯一不知道的是第 N 轮的**采样结果**——也就是请求是否在第 N 轮遇到了 EOS。所以一个在第 N 轮遇到 EOS 的请求，会被多调度一轮。

**第三，两条 stream。** 调度器在自己的 stream 上准备元数据（把位置、下标等从锁页内存异步拷到 GPU），引擎在另一条 stream 上计算。发射前 `self.engine.stream.wait_stream(self.stream)` 让计算等元数据拷完；而调度器 stream 上下一轮的准备工作不必等当前的计算。`ForwardInput` 把这一轮用到的所有张量（包括下标）保存到处理结果之后，防止它们在 GPU 还没用完时被释放、内存被复用。

## 由此带来的四个问题

"CPU 状态提前一轮"和"请求可能被多调度一轮"这两点，让处理结果变得微妙。官方实现处理了其中一种情况（防止重复释放），本书在复刻时又发现了四个问题，示例和测试都能复现：

@@code examples/ch11_overlap.py@@

@@output ch11_overlap@@

### 问题一：结束标记提前了一个 token

官方用 `finished = not req.can_decode` 判断是否达到 `max_tokens`。但处理第 N 轮时，第 N+1 轮已经发射，`device_len` 已经被推进了一步，`can_decode` 反映的是"第 N+1 轮之后"的状态。于是 `max_tokens=3` 的请求在第 2 个 token 时就被标记为结束（上面的 `(13, True)`），第 3 个 token 又发了一条带结束标记的消息。

离线接口恰好不受影响（它记下了每一条消息的 token），但在线服务的前端收到第一条 `finished=True` 就结束响应，**客户端只会收到 `max_tokens - 1` 个 token**。修正方法是改用与调度节奏无关的量——CPU 上已经收到的 token 数：

```python
finished = len(req.input_ids) >= req.max_device_len
```

### 问题二：EOS 之后多发一条过期消息

一个在第 N 轮遇到 EOS 的请求会被多调度一轮（前提二）。官方用 `finished_reqs` 记录上一轮结束的请求，处理第 N+1 轮时跳过它们的**资源释放**，但仍然为它们生成了回复消息——上面的 `(6722, False)`。前端已经删除了这个请求，会忽略这条消息；detokenizer 却会为它新建一个解码状态，而且永远等不到结束，造成一点内存泄漏。修正：在循环开头直接跳过这些请求的一切处理。

@@code python/minisgl/scheduler/scheduler.py:Scheduler._process_last_data@@

### 问题三：请求槽过早复用

@@diagram overlap-hazard 请求在第 N 轮遇到 EOS 之后发生了什么@@

请求在第 N 轮遇到 EOS，处理第 N 轮结果时它的资源被释放，其中包括它在 page table / token pool 里占的那一行。但此刻第 N+1 轮正在 GPU 上跑，其中还有这个请求：它会在引擎 stream 上把采样结果写进 token pool 的这一行。如果下一轮调度立即把这一行分给新请求，调度器 stream 上对这一行的写入（新请求的提示词）就和引擎 stream 上的旧写入没有先后保证，新请求的输入可能被覆盖。

KV 页没有这个问题：新请求写 KV 的 batch 在引擎 stream 上一定排在第 N+1 轮之后。请求槽的问题在于它被**调度器 stream**写入。修正：重叠模式下，释放的请求槽推迟到"下一次处理结果"时才真正归还——那时 `copy_done.synchronize()` 已经保证在途的 batch 算完了。没有在途 batch 时（`last_data is None`）立即归还，否则只有一个请求槽的交互模式会卡住。

@@code python/minisgl/scheduler/scheduler.py:Scheduler._free_req_resources@@

### 问题四：prefill 在途时收到 abort

客户端断开连接会触发 abort。如果 abort 到达时，这个请求的 prefill 恰好已经发射、结果还没处理，官方的流程是：abort 释放请求（`cache_req(finished=True)`，提示词的 KV 进入 Radix Cache）；随后处理 prefill 结果时，请求没有结束，于是再调用一次 `cache_req(finished=False)`——对一个已经释放的请求。第二次插入发现前缀已在缓存中，把"重复的那份"释放掉，而这些页正是缓存里那份：**同一批页既在空闲列表里又在缓存树里**。之后它们可能被分给新请求，覆盖掉缓存的 KV；调度器空闲时的完整性检查也会报错（Radix Cache 下复现为 `free_pages(1024) + cache_pages(5) != num_pages(1024)`）。

修正只需一行：abort 时把请求加入 `finished_reqs`，在途 batch 的结果就会被问题二的逻辑整体丢弃。

!!! diff "与官方的差异"
    以上四处都是对官方重叠调度的修正，每一处都有对应的测试：把修正改回官方写法，测试就会失败。它们在 CPU 上的复现依赖"处理第 N 轮时第 N+1 轮已经发射"这个时序，与是否真的有 GPU 并发无关——问题三的测试直接检查"分配请求槽时，它不属于任何在途 batch 中的请求"这个不变量。

!!! upstream "官方实现"
    - @@upstream scheduler/scheduler.py:Scheduler.overlap_loop@@
    - @@upstream scheduler/scheduler.py:Scheduler._process_last_data@@（`finished = not req.can_decode`、`if finished and req not in self.finished_reqs`）
    - @@upstream scheduler/scheduler.py:Scheduler._process_one_msg@@（abort 分支）
    - 关闭重叠调度：环境变量 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`，官方 README 里的消融实验就用它

## 在 CPU 上意味着什么

CPU 上没有异步执行：`NullStream`、`NullEvent` 都是空操作，"发射"时计算就已经完成。所以在 CPU 上重叠调度不会更快，但**调度逻辑完全相同**，上面四个问题照样出现、照样被测试抓住。性能上的收益要在 GPU 上测：官方 README 建议用 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 做消融。模型越小、每步 GPU 时间越短，CPU 开销的占比越大，重叠的收益越明显。第 21 章给出在 GPU 上做这个消融的方法。

## 测试

@@code tests/test_ch11_overlap.py:test_overlap_equals_normal_and_hf@@

重叠与普通两种循环、三种配置组合（radix / naive、分块、page size 4），输出都与 Hugging Face 相同。另外四个测试分别对应上面的四个问题：

@@code tests/test_ch11_overlap.py:test_abort_while_prefill_is_in_flight@@

!!! interview "怎么讲清楚"
    讲重叠调度：先发射第 N+1 轮，再在 CPU 上处理第 N 轮的结果，调度、组批、反分词这些 CPU 开销就藏在 GPU 计算后面。前提有三：第 N+1 轮的输入 token 在 GPU 上由第 N 轮的采样结果直接写入（CPU 不必知道具体值）；请求状态在发射时提前推进；调度器和引擎用两条 stream。代价是 CPU 的状态领先 GPU 一轮：在第 N 轮采样出 EOS 的请求，可能已经被放进了第 N+1 轮，要丢弃它多算的结果；结束判断要用已经收到的 token 数，请求槽要等在途的 batch 算完再复用。收益在小模型、小 batch（GPU 一步很短）时最大。

## 练习

1. 在 `overlap_loop` 里，如果把"发射本轮"和"处理上一轮"的顺序对调回来，但保留 `last_data` 的传递方式，会得到什么？
2. 问题二里的"多调度一轮"浪费了一次计算。能不能在调度第 N+1 轮之前就知道第 N 轮有没有 EOS？代价是什么？
3. 为什么问题三只影响请求槽，而 KV 页可以立即复用？如果新请求命中了前缀缓存，调度器 stream 会往 page table 写入命中的位置，这会不会引出类似的问题？

??? success "参考答案"
    1. 等价于普通循环（发射后立即等待结果），失去重叠的收益，但逻辑仍然正确。
    2. 可以在发射第 N+1 轮之前 `synchronize` 第 N 轮的结果，但这正是重叠调度要避免的等待。另一种思路是接受这一轮浪费：一个请求最多多算一个 token，对吞吐的影响只和"每轮结束的请求数"成正比，通常可以忽略。
    3. KV 池只被引擎 stream 上的 kernel 写入，新请求的 KV 写入排在旧 batch 之后；请求槽对应的 token pool 行和 page table 行则会被调度器 stream 写入（新请求的提示词、命中的位置、新分配的页）。命中前缀时写的是新请求自己的那一行，只要这一行不是刚释放、仍被在途 batch 使用的，就没有问题——这正是推迟释放请求槽所保证的。

## 小结

- [x] 重叠调度先发射本轮、再处理上一轮，CPU 开销藏在 GPU 计算之后。
- [x] 前提：输入在 GPU 上由上一轮的采样结果写入；请求状态在发射时提前推进；调度器与引擎用两条 stream。
- [x] 代价：CPU 状态领先一轮，请求可能在 EOS 之后被多调度一轮。
- [x] 本书修正了官方的四个问题：结束判断改用已收到的 token 数；丢弃过期结果；请求槽推迟到在途 batch 算完再复用；abort 的请求加入 `finished_reqs`。
