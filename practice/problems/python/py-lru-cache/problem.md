---
title: LRU 缓存
chapter: core/containers.md
difficulty: 中等
tags: [OrderedDict, 哈希表, 设计]
---
实现一个容量有限的 **LRU（最近最少使用）缓存** `LRUCache`：

- `LRUCache(capacity)`：`capacity >= 1`；
- `get(key)`：键存在时返回值，并把它标记为"最近使用"；不存在时返回 `-1`；
- `put(key, value)`：写入或更新键值，也算一次"使用"；写入后如果超出容量，淘汰**最久没有被使用**的键；
- `len(cache)` 返回当前的键数，`key in cache` 判断键是否存在（**不算**一次使用）。

`get` 和 `put` 都要求平均 $O(1)$。

```python
cache = LRUCache(2)
cache.put(1, "a")
cache.put(2, "b")
cache.get(1)        # "a"，此时 2 成了最久没用的
cache.put(3, "c")   # 淘汰 2
cache.get(2)        # -1
len(cache)          # 2
```

推理引擎里到处是 LRU：前缀缓存的淘汰、KV 卸载的换出，用的都是这个思路。

<!-- 题解 -->
`collections.OrderedDict` 同时是哈希表和双向链表：`move_to_end(key)` 把键移到末尾（最近使用），
`popitem(last=False)` 弹出开头（最久没用），两者都是 $O(1)$。`__contains__` 直接查字典，不移动顺序。

如果不用 OrderedDict，就要自己写"哈希表 + 双向链表"：字典存 key → 链表节点，链表按使用时间排序。
