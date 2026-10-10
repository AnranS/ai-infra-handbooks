# Radix Cache：跨请求复用前缀

<p class="lead">多轮对话里，第 N 轮的提示词包含前 N−1 轮的全部内容；很多应用给每个请求加上同一段系统提示词；Agent 每一步都带着越来越长的历史。这些请求的开头大段相同，它们的 KV 也完全相同。SGLang 的 RadixAttention 用一棵基数树记住所有算过的前缀，新请求来了先在树上找最长的匹配，命中的部分直接复用。这是 SGLang 最有代表性的设计，mini-sglang 用 237 行实现了它。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 基数树（radix tree）和普通的前缀树（trie）有什么区别？
    2. 一个节点什么时候可以被淘汰？为什么只淘汰叶子？
    3. 匹配前缀会修改树吗？
    4. page size 为 16 时，一个 30 个 token 的提示词最多能被缓存多少个 token？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 前缀树每个节点只存一个 token；基数树把没有分叉的一串 token 压缩进一个节点，节点保存一段 token 和它们的 KV 位置，匹配时一次比较一整段，节点数少得多。
    2. 没有被正在运行的请求锁住（`ref_count == 0`）、并且是叶子时。内部节点是别的节点的前缀，淘汰它会让子节点变得不可达；只淘汰叶子，树始终是合法的前缀结构（叶子被淘汰后，它的父节点可能成为新的叶子）。
    3. 会：匹配结束在一个节点中间时，要把这个节点分裂成两段，让句柄精确指向匹配结束的位置。
    4. 16 个：插入时只缓存完整的页，30 个 token 只有第一页（前 16 个）进入缓存。

**本章要写的文件**：`kvcache/radix_cache.py`、`kernel/torch_ops.py` 中的 `fast_compare_key`；在 `kvcache/__init__.py` 里注册 `radix`。

@@tree@@

这一步的 main：`examples/ch09_radix.py`——它只用到上面这些文件；`python tools/steps.py check` 会逐章搭出这棵树、跑这个 main。

@@video radix 动画：Radix Cache 的插入、分裂、匹配、加锁与淘汰（约 2 分钟）@@

## 树的结构

树的每个节点保存一段 token（`key`）和这些 token 的 KV 在池中的位置（`value`，与 page table 一样是 token 级位置）。从根走到某个节点，经过的所有 key 拼起来就是一个被缓存的前缀，对应的 value 拼起来就是它的 KV 位置。

和逐 token 建节点的前缀树（trie）不同，基数树把"没有分叉的一串"压缩成一个节点。一段 4000 token 的系统提示词在 trie 里是 4000 个节点，在基数树里是一个节点，匹配时一次比较一整段。

@@code python/minisgl/kvcache/radix_cache.py:RadixTreeNode@@

- `children` 按孩子 key 的**第一个 token**（page size 为 1 时）或**第一页的 token 元组**索引：同一个节点下，两个孩子的开头必然不同，所以一次字典查找就能找到可能匹配的孩子；
- `ref_count`：有多少个运行中的请求依赖这个节点。大于 0 时不能淘汰；
- `timestamp`：最近一次被访问的时间，用于 LRU 淘汰；
- `split_at(pos)`：把节点在 `pos` 处一分为二，新建的前半段插到父节点和自己之间。

## 在树上走

匹配和插入都从同一个函数开始：

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache._tree_walk@@

从根出发，用输入剩余部分的开头查孩子；找到了就比较这个孩子的 key 与输入有多长的公共前缀（`fast_compare_key`，按页向下对齐）。完全匹配就走下去继续；只匹配了一部分，就在匹配处把节点分裂，返回前半段。

所以**匹配会改变树的形状**：它不增删任何缓存内容，但可能把一个节点拆成两个，好让返回的句柄精确地指向"匹配结束的地方"。这对后面的加锁很重要——锁住的正好是命中的部分，不多不少。

`fast_compare_key` 找第一个不同元素的位置，官方用 C++ 的 `std::mismatch` 实现（通过 tvm-ffi 绑定），我们用 PyTorch：

@@code python/minisgl/kernel/torch_ops.py:fast_compare_key@@

## 匹配、插入、加锁

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.match_prefix@@

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.insert_prefix@@

插入只缓存完整的页（`align_down`）：page size 为 16 时，30 个 token 的提示词只有前 16 个进入缓存。走到已有前缀的尽头后，把剩下的部分挂成一个新叶子。返回值里的 `cached_len` 是"插入之前就已经在树里的长度"，调用方（`CacheManager.cache_req`）据此释放自己那份重复的页。

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.lock_handle@@

加锁沿着句柄节点一直走到根，路径上每个节点的 `ref_count` 加一——请求依赖的是整条前缀，不只是最后一段。节点的 `ref_count` 从 0 变成 1 时，它的长度从"可淘汰"移到"受保护"。这两个计数加起来就是缓存里的总 token 数；`CacheManager.available_size` 用的正是"空闲页 + 可淘汰"。

## 淘汰

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.evict@@

只有 `ref_count` 为 0 的**叶子**能淘汰：淘汰中间节点会让它的孩子失去前缀，变得不可达。把所有可淘汰的叶子按时间戳放进一个小顶堆，每次弹出最久没被访问的；它的父节点如果因此变成了叶子、且没有被锁，也加入堆中。淘汰以节点为单位，所以实际释放的量可能比要求的多。

## 看一个例子

@@diagram radix-tree 插入、匹配（分裂与加锁）、LRU 淘汰@@

@@code examples/ch09_radix.py@@

@@output ch09_radix@@

- 插入 `[1,2,3,4]` 和 `[1,2,5,6]` 后，公共前缀 `[1,2]` 成为一个节点，下面分出两支；
- 匹配 `[1,2,3,9]` 命中 3 个 token：`[3,4]` 被分裂成 `[3]` → `[4]`，句柄指向 `[3]`。加锁后 `[1,2]` 和 `[3]` 的 `ref_count` 变为 1，受保护 3 个 token，可淘汰 5 个；
- 要求淘汰 2 个 token：可淘汰的叶子是 `[4]`、`[5,6]`、`[7,8]`。匹配只刷新了 `[1,2]` 和 `[3]` 的时间戳，分裂出的 `[4]` 保留了最初插入时的时间，最先被淘汰；1 个不够，接着淘汰第二早的 `[5,6]`，实际释放了 3 个；
- 再淘汰 1 个 token：只剩 `[7,8]` 可以淘汰。被锁住的 `[1,2]`、`[3]` 留了下来。

端到端的效果：4 个请求共享一段系统提示词，第一个请求完整地 prefill，后面三个只算各自的问题部分——prefill 的计算量从 231 个 token 降到 73 个。

## 与请求生命周期的配合

回顾第 8 章的 `cache_req`，现在它的四段都真实存在了：

1. 接纳时 `match_req` 找到命中的前缀并加锁（匹配到倒数第二个 token 为止，保证至少算一个 token）；
2. prefill 结束时 `cache_req(finished=False)`：把整个提示词（按页对齐）插入树中，解锁旧句柄、锁住新句柄；如果期间别的请求已经插入了相同的前缀，自己那份重复的页立即释放；
3. 请求结束时 `cache_req(finished=True)`：把生成的内容也插入树中（下一轮对话会用到），解锁，释放不足一页的尾巴。

请求结束后它的 KV 并没有被释放，而是留在树里变成"可淘汰"；只有在需要空间时才真正被淘汰。所以显存总是被尽量用来缓存前缀。

!!! upstream "官方实现"
    - 整个文件：@@upstream kvcache/radix_cache.py@@
    - 树上行走：@@upstream kvcache/radix_cache.py:RadixPrefixCache._tree_walk@@
    - 淘汰：@@upstream kvcache/radix_cache.py:RadixPrefixCache.evict@@
    - C++ 实现的 `fast_compare_key`：`kernel/csrc/src/radix.cpp`

!!! diff "与官方的差异：完整性检查"
    官方的 `RadixPrefixCache.check_integrity` 是空的（`pass`）。我们的版本遍历整棵树，重新统计可淘汰和受保护的长度，与计数器核对，并检查"孩子被锁的次数不超过父亲"这个不变量。调度器空闲时调用它，任何一处计数错误都会立即暴露。

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.check_integrity@@

## 测试

@@code tests/test_ch09_radix.py:test_lock_protects_from_eviction_and_lru_order@@

@@code tests/test_ch09_radix.py:test_shared_prefixes_are_reused_and_outputs_unchanged@@

同样的 5 个提示词跑第二遍，每个请求只需重算最后 1 个 token，输出与 Hugging Face 完全相同；内存完整性检查通过。

!!! interview "怎么讲清楚"
    讲前缀缓存：基数树是压缩的前缀树，一个节点保存一段 token 和它们的 KV 位置，孩子按第一个 token（或第一页）索引，匹配时一次比较一整段；匹配可能在节点中间结束，这时分裂节点让句柄精确指向匹配处（所以匹配也会修改树）。只缓存完整的页（页大小 16 时 30 个 token 最多缓存 16 个）。正在使用的请求沿路径加锁（引用计数加一），只有未被锁住的叶子能按 LRU 淘汰；请求结束后 KV 留在树里，需要空间时才淘汰。和 vLLM 的哈希块前缀缓存相比：基数树按任意长度匹配、天然支持多轮对话，哈希块按整块查表、实现简单、便于分布式和多级缓存。

## 练习

1. vLLM 用"哈希块"做前缀缓存：每个 KV 块的哈希值由"前一个块的哈希 + 本块的 token"算出，用一个字典查找。和基数树相比，各有什么优缺点？（参考[推理系统手册的前缀缓存一章](serving://engine/prefix-cache/)。）
2. 现在的淘汰每次都遍历整棵树收集可淘汰的叶子。当树很大（几十万个节点）时这是一个瓶颈。怎样维护一个始终有效的"可淘汰叶子集合"？
3. 设计一个测试，构造"两个前缀相同的请求在同一个 batch 里 prefill"的场景，验证 `cache_req` 正确释放了重复的页。

??? success "参考答案"
    1. 哈希块：实现简单，查找是 O(块数) 次字典访问，天然按块对齐，易于跨进程共享（KV 事件）；但每次都要从头逐块哈希，且只能按块粒度匹配。基数树：一次比较一整段，共享前缀在树上只存一份，淘汰时天然知道依赖关系；但实现复杂，节点分裂合并需要小心维护计数。
    2. 维护一个集合（或按时间戳排序的堆，配合懒删除）：节点变成叶子且 `ref_count` 为 0 时加入；被加锁、有了孩子、或被淘汰时移除。SGLang 0.5.20 的 `RadixCache` 正是这样做的：它维护一个 `evictable_leaves` 集合，在每次加锁、解锁、插入、删除后用 `_update_leaf_status` 更新。
    3. 用 `PROMPTS` 里的 "The capital of France is" 和 "The capital of France is a city that"：它们在同一个 prefill batch 里都没有命中；先结束的插入前缀后，后结束的插入时 `cached_len > 0`，走"释放重复页"的分支。跑完后 `check_integrity()` 通过即说明没有泄漏——`test_shared_prefixes_are_reused_and_outputs_unchanged` 已经覆盖了这个场景。

## 小结

- [x] 基数树的节点保存一段 token 和它们的 KV 位置；孩子按第一个 token（或第一页）索引，匹配时一次比较一整段。
- [x] 匹配可能分裂节点，让句柄精确指向匹配结束处；插入只缓存完整的页，把新部分挂成叶子。
- [x] 加锁把整条路径的 `ref_count` 加一；只有未被锁的叶子能按 LRU 淘汰，淘汰以节点为单位。
- [x] 请求结束后 KV 留在树里，需要空间时才淘汰；共享系统提示词时 prefill 计算量大幅下降。
