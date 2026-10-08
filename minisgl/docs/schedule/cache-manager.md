# CacheManager：分配、释放与准入控制

<p class="lead">KV 池的每一页，在任何时刻只有三种状态：空闲、被某个请求独占、或者在前缀缓存里（可能同时被若干请求共享）。<code>CacheManager</code> 负责在这三种状态之间搬运页，并回答调度器最关心的问题：再接纳一个请求，KV 还放得下吗？mini-sglang 的答案很保守——按最坏情况预留，所以它永远不需要抢占。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. vLLM 在 KV 不够时会抢占（preempt）正在运行的请求，mini-sglang 为什么不需要？代价是什么？
    2. `available_size` 为什么要把前缀缓存里"可淘汰"的部分也算进去？
    3. 匹配前缀时为什么只匹配到提示词的倒数第二个 token？
    4. page size 为 4 时，一个请求 `cached_len=6`、`device_len=9`，本轮要分配几页？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 准入控制按最坏情况（剩余提示词 + `max_tokens`）预留 KV，接纳的请求一定能跑完，永远不需要抢占；代价是预留的空间大多用不满，同时运行的请求少，并发较低。
    2. 前缀缓存里没有被锁住的节点随时可以淘汰腾出空间，它们占着显存，但对新请求来说是"可用"的；不算进去会低估可用空间，白白拒绝请求。
    3. 每一轮至少要算一个 token：即使整个提示词都命中，也需要最后一个位置的隐状态来算 logits，所以匹配到倒数第二个 token 为止，保证 `cached_len < device_len`。
    4. 要为 `[ceil(6/4), ceil(9/4)) = [2, 3)` 这些页分配，即 1 页：位置 6、7 还在已经分配的第 1 页里（覆盖位置 4～7），只有位置 8 需要新的第 2 页。

**本章要写的文件**：`scheduler/cache.py`；补全 `scheduler/prefill.py` 中的 `PrefillAdder._try_allocate_one`。

## 空闲页列表

@@code python/minisgl/scheduler/cache.py:CacheManager.__init__@@

`free_slots` 存的是每个空闲页**第一个 token 的位置**（page size 为 4 时是 `[0, 4, 8, …]`），而不是页号。这与第 4 章"page table 按 token 存位置"一致：分配出去的页只需加上页内偏移就是 token 位置，释放时从 token 位置里每隔 page size 取一个就回到页的起点。

@@code python/minisgl/scheduler/cache.py:CacheManager._allocate@@

分配就是从 `free_slots` 前面切一段。空闲页不够时，先让前缀缓存淘汰一些（第 9 章），把淘汰出来的页补进来再分配。

## 为本轮的 token 分配页

@@code python/minisgl/scheduler/cache.py:CacheManager.allocate_paged@@

对每个请求，已经分配过的页是 `[0, ceil(cached_len / page_size))`，本轮结束后需要的页是 `[0, ceil(device_len / page_size))`，差值就是要新分配的页。page size 为 4、`cached_len=6`、`device_len=9` 时：已有 2 页（覆盖位置 0～7），需要 3 页，新分配 1 页。decode 时大多数轮次不需要新页，每 page size 轮才分配一次。

所有请求需要的页一次分配，再一次性写进 page table：

@@code python/minisgl/scheduler/cache.py:_write_page_table@@

先在 CPU 上算好所有 `(行, 列)` 下标，再用一次索引赋值写入。GPU 上这是一次异步拷贝加一个 kernel，而不是每个请求、每个 token 一次操作。

## 释放与"懒释放"

@@code python/minisgl/scheduler/cache.py:CacheManager.lazy_free_region@@

处理一个 batch 的结果时，可能有很多请求同时结束，每个都要释放若干页。如果每次都 `torch.cat` 一次 `free_slots`，就是 N 次拷贝。`lazy_free_region` 在这个区域内把 `_free` 换成"先记下来"，退出时一次性拼接。调度器的 `_process_last_data` 整个都包在这个区域里。

## 准入控制

第 7 章开头的动画后半段演示了下面这套准入规则：按最坏情况预留，所以永不抢占。

请求能不能被接纳，由 `PrefillAdder._try_allocate_one` 决定：

@@code python/minisgl/scheduler/prefill.py:PrefillAdder._try_allocate_one@@

判断标准是**最坏情况**：这个请求还需要的 KV = 没命中缓存的提示词 + `max_tokens`。加上 `reserved_size`（已经承诺给别人的），不能超过 `available_size`（空闲页 + 前缀缓存里可以淘汰的部分）。

`reserved_size` 有两个来源：

- 组这个 batch 之前，正在 decode 的请求未来还可能用到的空间——`DecodeManager.inflight_tokens`，每个请求的 `remain_len` 之和，再给每个请求多留一页的余量（最后一页可能只用了一部分）；
- 本 batch 里已经接纳的请求（`_add_one_req` 里 `reserved_size += remain_len + output_len`）。

检查做了两次，中间是 `lock(handle)`：加锁之前，命中的前缀在前缀缓存里是"可淘汰"的，被算在 `available_size` 里；加锁之后它变成"受保护"，`available_size` 相应变小。第一次检查失败可以直接返回，省掉加锁再解锁的开销；第二次检查才是准确的。

因为每个请求在接纳时就预留了最坏情况的空间，运行中的请求永远不会因为 KV 不够而停下，mini-sglang 不需要抢占（preemption）和重算。代价是保守：`max_tokens` 设得很大而实际很早遇到 EOS 的请求，预留的空间大部分没用上，能同时运行的请求变少。vLLM 采取相反的策略——乐观地接纳，KV 不够时抢占最晚到达的请求、释放它的 KV，之后重算。

看一个 KV 池只有 64 个 token 的例子：

@@code examples/ch08_cache_manager.py@@

@@output ch08_cache_manager@@

前 4 个请求最坏情况共需 13 + 13 + 12 + 23 = 61 个 token，放得下；第 5 个还需要 13 个，超过 64，只能等待。可以看到直到前 4 个请求全部结束，池里始终有空闲页（最后还剩 7 页）——这就是保守预留的代价。前 4 个结束后空间归还，第 5 个才被接纳。全部完成后 64 页全部回到空闲列表，完整性检查通过。

## 把 KV 交给前缀缓存

请求的 KV 什么时候、以什么方式交给前缀缓存，是 `CacheManager` 里最精细的一段逻辑：

@@code python/minisgl/scheduler/cache.py:CacheManager.cache_req@@

它在两个时刻被调用：prefill 刚结束时（`finished=False`，提示词的 KV 进入缓存，请求继续运行），和请求结束时（`finished=True`）。注释里把请求的 KV 分成了四段：

@@diagram cache-regions cache_req 把请求的 KV 分成四段处理@@

中间那段"自己算的、但缓存里已经有了"是怎么来的？两个前缀相同的请求同时到达、同时 prefill，谁都没有命中缓存；先结束的那个把前缀插进了缓存，后结束的那个再插入时发现已经存在——它自己算的那份就是多余的，必须释放，否则就泄漏了。naive 缓存下 `insert_prefix` 什么都不保留，于是请求的页在结束时全部释放。第 9 章实现 Radix Cache 后，这四段都会真实出现。

## 完整性检查

@@code python/minisgl/scheduler/cache.py:CacheManager.check_integrity@@

调度器空闲时（`run_when_idle`）做一次检查：此时没有运行中的请求，所以每一页要么空闲、要么在缓存里，两者之和必须等于总页数。任何一处忘了释放（或者重复释放），都会在这里暴露出来。

!!! upstream "官方实现"
    - @@upstream scheduler/cache.py:CacheManager@@
    - 准入控制：@@upstream scheduler/prefill.py:PrefillAdder._try_allocate_one@@
    - 预留：@@upstream scheduler/decode.py:DecodeManager.inflight_tokens@@

    官方仓库的 `tests/core/test_cache_allocate.py` 专门测试了 page size 大于 1 时"先淘汰、再分配"得到的页是否按页对齐、互不重叠。

## 测试

@@code tests/test_ch08_cache_manager.py:test_allocate_paged_writes_page_aligned_token_positions@@

@@code tests/test_ch08_cache_manager.py:test_integrity_check_detects_leak@@

!!! interview "怎么讲清楚"
    讲 KV 管理：空闲页列表按页存起始位置，每轮只为 `[ceil(cached_len / 页大小), ceil(device_len / 页大小))` 这些页分配，decode 每"页大小"轮才分配一次；不够时让前缀缓存淘汰，所以可用空间要把缓存里可淘汰的部分算进去；释放批量延迟做（lazy free）。准入控制按最坏情况（剩余提示词 + max_tokens）预留，所以永远不需要抢占，代价是并发比 vLLM 低（vLLM 乐观接纳、不够时抢占重算）。匹配前缀只匹配到提示词的倒数第二个 token，保证至少算一个 token 拿到 logits。请求结束时 KV 分段交给前缀缓存，每一页有且只有一个归属，空闲时的完整性检查能抓住泄漏。

## 练习

1. `inflight_tokens` 给每个运行中的请求多预留 `page_size - 1` 个 token。举一个例子说明，如果不预留，page size 为 16 时会出什么问题。
2. 实现一个"乐观"的准入策略：只预留 `max_tokens` 的一半，KV 不够时抢占最后接纳的请求（释放它的资源、放回等待队列队首）。需要改哪些地方？被抢占的请求重新 prefill 时，前缀缓存能帮上什么忙？
3. 如果 `cache_req` 忘了释放 `[old.cached_len, cached_len)` 这一段，会在哪里被发现？

??? success "参考答案"
    1. 请求的 KV 按页分配：`remain_len` 只按 token 计算，但下一次分配是一整页。例如某请求 `device_len=17`、`remain_len=1`，需要的却是一个完整的新页（16 个 token）。多个这样的请求加起来，按 token 估算会低估实际需要的页数，分配时可能失败。
    2. 在 `_prepare_batch` 的 `allocate_paged` 之前检查空闲页，不够时从 decode 集合中挑一个请求：`table_manager.free`、`cache_manager.cache_req(finished=True)`，把它包装回 `PendingReq` 放到队首。它已经生成的 token 要拼进 `input_ids`。因为 `cache_req` 把它的 KV 交给了 Radix Cache，重新 prefill 时大部分前缀会直接命中（除非已被淘汰），重算的代价很小——这正是 SGLang 正式版的做法。
    3. 这些页既不在空闲列表里，也不属于前缀缓存（缓存里存的是另一份相同内容的页），请求结束后就没有任何人持有它们。调度器下一次空闲时的 `check_integrity` 会发现"空闲页 + 缓存页 ≠ 总页数"并报错。第 9 章的端到端测试在两个相同前缀的请求同时 prefill 的场景下会覆盖到这条路径。

## 小结

- [x] `free_slots` 按页存起始位置；分配从前面切，不够时让前缀缓存淘汰；释放在 `lazy_free_region` 里批量完成。
- [x] 每轮只为 `[ceil(cached_len/ps), ceil(device_len/ps))` 这些页分配，decode 每 page size 轮才分配一次。
- [x] 准入控制按最坏情况（剩余提示词 + max_tokens）预留，因此永不抢占；代价是并发较低。
- [x] `cache_req` 把请求的 KV 分成四段处理，保证每一页都有且只有一个归属；空闲时的完整性检查能抓住任何泄漏。
