---
title: 哈希块前缀缓存（带 LoRA 隔离）
chapter: engine/prefix-cache.md
difficulty: 困难
tags: [前缀缓存, 链式哈希, LRU]
---
实现正文的 vLLM 式前缀缓存 `PrefixCachingBlockPool(num_blocks, block_size)`，并加一个真实系统里必须有的功能：**按 `extra_key` 隔离**。
同样的 token 用不同的 LoRA 适配器（或不同的图片）算出来的 KV 不同，不能互相命中：块哈希链的**第一个块**要把 `extra_key` 混进去。

需要实现的方法（语义与正文相同）：

| 方法 | 说明 |
| --- | --- |
| `allocate(n)` | 从空闲队列**头部**取 `n` 个块（不够返回 `None`），引用计数置 1；取到的块如果带着哈希，就把它从缓存里驱逐 |
| `free(blocks)` | **倒序**把引用计数减 1，减到 0 的块放到空闲队列**尾部**（内容和哈希保留，还能被命中） |
| `touch(blocks)` | 命中的块引用计数加 1，如果它在空闲队列里就移出来 |
| `block_hashes(token_ids, extra_key=None)` | 只对装满的块计算链式哈希：`h_0 = hash((extra_key, tuple(块0)))`，`h_i = hash((h_{i-1}, tuple(块i)))` |
| `lookup(token_ids, extra_key=None)` | 返回最长的已缓存前缀对应的块列表；同时累加统计 `num_queries`（查询的 token 数）和 `num_hits`（命中的 token 数） |
| `cache_blocks(token_ids, block_ids, num_computed, extra_key=None)` | 把 `token_ids[:num_computed]` 里装满的块登记进缓存（块已经有哈希、或者这个哈希已经被别的块登记过时跳过） |

另外实现 `hit_rate()`：`num_hits / num_queries`（没有查询时为 0.0）。

<!-- 题解 -->
空闲队列用 `collections.OrderedDict`：`popitem(last=False)` 从头部取、`q[b] = None` 放到尾部、`del q[b]` 从中间移出，都是 $O(1)$。

链式哈希保证"哈希相同 ⇔ 从开头到这个块的所有 token（以及 extra_key）都相同"。extra_key 只需要混进第一个块：后面的块通过父哈希间接包含了它。
没有 extra_key 时 `h_0 = hash((None, 块0))`，与正文 `parent = None` 的写法一致。
