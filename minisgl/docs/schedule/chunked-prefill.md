# 分块 prefill

<p class="lead">一个 3 万 token 的提示词如果一次 prefill，激活值（尤其是 MLP 的中间结果和 logits 之前的隐状态）可能直接撑爆显存，而且这一轮会占用 GPU 很久，其他请求都得等着。分块 prefill（chunked prefill，出自 Sarathi-Serve）把长提示词切成若干块，每轮只算一块。mini-sglang 的实现非常紧凑：一个"不能 decode 的请求"子类 <code>ChunkedReq</code>，加上 <code>PrefillAdder</code> 里的几行预算逻辑。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 分块 prefill 的结果与一次性 prefill 在数学上等价吗？为什么？
    2. 一个请求被切成了 4 块，第 2 块要算注意力时，前面 16 个 token 的 KV 从哪来？
    3. 分块中的请求这一轮产生的 logits 要不要采样？
    4. mini-sglang 里，一个请求正在分块 prefill 时，其他正在 decode 的请求会怎样？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 等价：因果注意力里每个 token 只依赖它之前的 token，后面的块通过页表读取前面块已经写好的 KV，得到的结果和一次算完相同（数值上只有浮点顺序的微小差异）。
    2. 从 KV 池里：前面几块算完时它们的 KV 已经写入池中，页表记着位置，第 2 块的注意力后端按页表读出来——这和命中前缀缓存是同一种情形。
    3. 不要：只有最后一块算完时，最后位置的 logits 才用来采样第一个输出 token；中间块的 logits 直接丢弃。
    4. 暂停：mini-sglang 一轮只做 prefill 或 decode，分块 prefill 期间 decode 要等；正式版的 SGLang 把分块 prefill 和 decode 混在同一个 batch 里解决这个问题。

**本章要写的文件**：补全 `scheduler/prefill.py` 中的 `ChunkedReq`、`PrefillAdder._add_one_req`、`PrefillAdder.try_add_one`。

## 为什么等价

注意力是因果的：第 j 个位置只依赖位置 ≤ j 的 token。把提示词切成 `[0,16)`、`[16,32)`、……，第一块算完后它的 KV 已经写进了 KV 池；第二块的 query 通过 page table 读取前 32 个位置的 KV（16 个来自上一块、16 个是本块刚写入的），与一次性 prefill 时第 16～31 个位置看到的完全相同。MLP、RMSNorm 都是逐位置独立的，更不受影响。所以分块 prefill 与一次性 prefill 在数学上等价，只是换了一种计算顺序。

这正是第 5 章"前缀命中"的同一种情形：对注意力后端来说，"前 16 个 token 在缓存里、本轮算 16 个"和"前 16 个 token 命中了前缀缓存"没有任何区别。所以分块 prefill 不需要注意力后端做任何改动。

## ChunkedReq

@@code python/minisgl/scheduler/prefill.py:ChunkedReq@@

分块中的请求是 `Req` 的一个子类，只改了两处：不能被采样（本轮 LM head 算出的 logits 属于提示词的中间位置，没有意义），也不能进入 decode 集合。

## 预算

@@code python/minisgl/scheduler/prefill.py:PrefillAdder._add_one_req@@

每个 prefill batch 有一个 token 预算 `token_budget`（即 `max_extend_tokens`，命令行 `--max-prefill-length`，默认 8192）。接纳一个请求时，本块的大小是"预算剩余"和"请求剩余"中较小的那个；如果本块没能覆盖请求剩下的全部提示词，就创建 `ChunkedReq`，否则创建普通的 `Req`。

`reserved_size` 按请求**剩余的全部**提示词加 `max_tokens` 预留，而不只是本块——请求第一次被接纳时，它未来需要的全部空间就已经承诺了，后续的块不会因为空间不够而卡住。

@@code python/minisgl/scheduler/prefill.py:PrefillAdder.try_add_one@@

`PendingReq.chunked_req` 记住分块进行到哪里。下一轮再遇到这个请求时，直接沿用它已经分配好的请求槽和缓存句柄，从上次的 `cached_len` 继续，不再做前缀匹配和准入检查。

回到上一章的 `PrefillManager.schedule_next_batch`：没做完的分块请求被放回等待队列的**队首**，下一轮优先继续：

```python
self.pending_list = chunked_list + self.pending_list[len(reqs):]
```

调度器处理结果时跳过 `ChunkedReq`（第 7 章 `_process_last_data` 开头的 `isinstance` 判断），也不把它交给前缀缓存；写回 token pool 时它的列号是 −1（不需要把采样结果当作下一轮输入）。

## 看一个例子

@@diagram chunked-prefill max_extend_tokens=16 时三个请求的 prefill 轮次@@

@@code examples/ch10_chunked.py@@

@@output ch10_chunked@@

61 个 token 的长提示词被切成 16 + 16 + 16 + 13；第四轮剩下 3 个 token 的预算，分给了排在后面的 uid1，它也成了一个分块请求；第五轮 uid1 做完剩下的 2 个 token，uid2 一次做完。三个请求都完成 prefill 后才开始 decode。最终输出与 Hugging Face 整段 prefill 的结果相同。

## 代价

注意上面的轨迹：在长提示词分块 prefill 的整个过程中，没有任何 decode。这是"prefill 优先"策略的直接结果——只要等待队列不空，调度器就做 prefill，而分块中的请求始终在队首。如果此时有别的请求正在 decode，它们的下一个 token 要等所有块都做完才会出来，TPOT 出现一个尖峰。

SGLang 正式版和 vLLM 的做法是**把 prefill 块和 decode 请求放进同一个 batch**：每轮 decode 请求照常各算一个 token，剩余的预算给 prefill 块。这样长提示词不会阻塞 decode，代价是 batch 里同时有两种形状，CUDA Graph 只能覆盖纯 decode 的轮次，调度逻辑也更复杂。mini-sglang 为了简单选择了"一轮只做一种"。

!!! upstream "官方实现"
    - @@upstream scheduler/prefill.py:ChunkedReq@@
    - @@upstream scheduler/prefill.py:PrefillAdder._add_one_req@@
    - 官方文档 docs/features.md 提醒：块太小（例如 128）会明显降低性能——每块都要完整地过一遍模型，权重被重复读取，块越小 GPU 越吃不满。

## 测试

@@code tests/test_ch10_chunked.py:test_chunked_prefill_is_exact_and_bounded@@

每个 prefill batch 恰好用满 6 个 token 的预算（最后一个除外），所有提示词 token 恰好各算一次，输出与 HF 一致。第 11 章的测试还会在"分块 + 重叠调度 + page size 4"的组合下再验证一遍。

!!! interview "面试怎么答"
    分块 prefill 题：因果注意力让分块和一次性 prefill 在数学上等价——后面的块通过 page table 读前面块已经写好的 KV，和命中前缀缓存是同一种情形；中间的块不采样（logits 丢掉），最后一块才采样并进入 decode。每个 prefill batch 受 `max_extend_tokens` 限制，未完成的分块请求回到队首优先继续。作用是限制单轮的时长和显存峰值；但 mini-sglang 一轮只做一种，分块期间 decode 仍会暂停，正式的 SGLang / vLLM 把分块 prefill 和 decode 混在同一个 batch 里，decode 每一步都能前进，只是变慢一些。

## 练习

1. 实现"混合 batch"：在 `_schedule_next_batch` 里，如果 decode 集合不空，先把所有 decode 请求放进 batch（每个 1 个 token），剩余的预算再给 prefill。`Batch.phase` 该怎么设？注意力后端和 LM head 需要改什么？
2. 分块请求被 abort（客户端断开）时，`PrefillManager.abort_req` 返回了它的 `chunked_req`，调度器随后调用 `_free_req_resources`。为什么非分块、还在等待的请求 abort 时不需要释放资源？
3. `max_extend_tokens` 设为 1 会发生什么？

??? success "参考答案"
    1. 可以沿用 `phase="prefill"`（因为有 query 数大于 1 的请求），把 decode 请求也当作 `extend_len = 1` 的"prefill"：注意力后端的元数据本来就支持任意的 `cu_seqlens_q`；LM head 用 `get_last_indices` 取每个请求的最后位置，对 decode 请求就是它唯一的位置。要改的主要是调度器的记账（decode 请求走 `_process_last_data` 的正常路径）和 CUDA Graph 只在纯 decode 时使用。
    2. 还没被接纳的请求没有分配任何东西（请求槽、KV 页、缓存锁都是在 `_try_allocate_one` 里才获得的），从队列里删掉即可；分块中的请求已经拿到了这些资源，必须释放。
    3. 每轮 prefill 只算 1 个 token，长提示词要跑和它长度一样多的轮次，每轮都要读一遍全部权重，极慢；但结果依然正确。

## 小结

- [x] 因果注意力让分块 prefill 与一次性 prefill 等价：后面的块通过 page table 读取前面块的 KV，与前缀命中是同一种情形。
- [x] `ChunkedReq` 不采样、不进 decode；`PendingReq.chunked_req` 记住进度，下一轮沿用资源继续。
- [x] 每个 prefill batch 受 `max_extend_tokens` 限制；分块请求回到队首优先继续。
- [x] mini-sglang 一轮只做一种，长提示词分块期间 decode 会暂停；正式版用混合 batch 解决。
