---
title: Radix Cache：匹配、插入、加锁、淘汰
chapter: schedule/radix-cache.md
difficulty: 困难
tags: [基数树, 前缀缓存, LRU]
---
用纯 Python 复刻书中的 `RadixPrefixCache`（`page_size = 1`，token 和 KV 位置都用列表）。为了让结果可复现，时间戳用一个逻辑时钟：**每次调用 `match_prefix` 或 `insert_prefix` 时时钟加 1**，这次调用里访问到的节点都刷新成这个值，新建的节点也用这个值。

节点：`key`（一段 token）、`value`（对应的 KV 位置）、`children`（按 key 的第一个 token 索引）、`parent`、`ref_count`、`timestamp`。实现 `RadixCache`：

| 方法 | 说明 |
| --- | --- |
| `match_prefix(ids)` | 从根往下走，返回 `(handle, matched_len)`。某个节点只匹配了一部分时，在匹配处**分裂**：前半段成为新的父节点（继承原节点的 `ref_count` 和时间戳，然后刷新为本次时钟），句柄指向前半段 |
| `insert_prefix(ids, indices)` | 先像 `match_prefix` 一样走（会分裂、刷新时间戳），走到尽头后把剩下的部分挂成一个新叶子。返回 `(prefix_len, handle)`：`prefix_len` 是插入之前就在树里的长度，`handle` 指向覆盖全部 `ids` 的节点 |
| `lock(handle)` / `unlock(handle)` | 从句柄节点一直到根（不含根），每个节点 `ref_count` 加 / 减 1；`ref_count` 在 0 和 1 之间变化时，把节点长度在 `evictable_size` 和 `protected_size` 之间挪动 |
| `evict(size)` | 淘汰至少 `size` 个 token：只能淘汰 `ref_count == 0` 的**叶子**，按时间戳从小到大（相同时按节点创建顺序）。父节点因此变成可淘汰的叶子时，也加入候选。返回被淘汰的 KV 位置（按淘汰顺序拼接）。`size` 超过 `evictable_size` 时抛出 `ValueError` |
| `matched_indices(handle)` | 从根到句柄节点的 `value` 拼起来 |

句柄 `handle` 有 `cached_len`（匹配/覆盖的长度）和 `node` 两个属性。`evictable_size`、`protected_size` 是属性（新插入的节点 `ref_count` 为 0，计入可淘汰）。

<!-- 题解 -->
和书中代码结构一致：

- `_walk(ids)`：`child = node.children.get(ids[pos])`；比较 `child.key` 和 `ids[pos:]` 的公共前缀长度 `m`；`m < len(child.key)` 时分裂并返回前半段；
- 分裂 `split_at(node, m)`：新建前半段节点，挂到原父节点下（替换 `children` 里的那一项），原节点的 key/value 截掉前 `m` 个，挂到新节点下；
- 淘汰：先收集所有 `ref_count == 0` 的叶子放进最小堆（按 `(timestamp, 创建序号)`），每弹出一个，从父节点的 `children` 里删掉它；父节点成了叶子且没被锁，就把它推进堆。

书中的例子：插入 `[1,2,3,4]`、`[1,2,5,6]`、`[7,8]`，匹配 `[1,2,3,9]` 并加锁，然后 `evict(2)` 释放 `[13, 20, 21]`，再 `evict(1)` 释放 `[30, 31]`——测试会复现它。
