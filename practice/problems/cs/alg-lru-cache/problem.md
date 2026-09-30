---
title: LRU 缓存
chapter: algo/linked-stack-hash.md
difficulty: 中等
tags: [LRU,哈希表,双向链表]
---
实现一个 O(1) 的 LRU 缓存 `LRUCache(capacity)`：

- `get(key)`：存在则返回值并把它标记为最近使用，否则返回 -1；
- `put(key, value)`：写入或更新，并标记为最近使用；超过容量时淘汰**最久未使用**的键；
- `keys_lru_first()`：返回当前所有键，按从最久未使用到最近使用的顺序（用于测试）。

`capacity` 为 0 时任何键都存不下。

```python
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
c.get(1)                 # 1
c.put(3, 3)              # 淘汰 2
c.get(2)                 # -1
c.keys_lru_first()       # [1, 3]
```

<!-- 题解 -->
两个结构配合：**哈希表**把 key 映射到节点（O(1) 定位），**双向链表**维护使用顺序（O(1) 摘除与插入）。链表一端是最久未使用、另一端是最近使用；`get` 命中就把节点挪到"最近"那一端，`put` 超容就从"最久"那一端摘掉。

Python 里可以直接用 `OrderedDict`：`move_to_end(key)` 对应"标记为最近使用"，`popitem(last=False)` 对应"淘汰最久未使用"。面试时要能说清楚手写版为什么需要**双向**链表（单向链表摘除节点时找不到前驱，退化成 O(n)）。

三个边界：`capacity == 0`、重复 `put` 同一个键（要更新值并挪位置，不能算成新键）、`get` 未命中不能插入。

推理引擎的块池就是这个结构：块哈希 → 块，空闲块按 LRU 排队；SGLang 的 Radix Cache 还加了引用计数，正在被使用的节点锁住不淘汰（见[前缀缓存](serving://engine/prefix-cache/)）。
