---
title: 页缓存：写回、淘汰与 fsync
chapter: os/io-stack.md
difficulty: 中等
tags: [页缓存, 写回, fsync, LRU]
---
实现一个简化的页缓存 `PageCache(capacity)`，最多缓存 `capacity` 个页，统计它对磁盘的读写次数：

- `read(page)`：页在缓存里是一次命中（`hits` 加 1）；不在就从盘上读（`disk_reads` 加 1），作为干净页放进缓存；
- `write(page)`：整页写入，不需要先读盘。页不在缓存里就放进去，然后标记为脏页；
- 缓存满了要放新页时，按 **LRU** 淘汰最久没被读写的页，淘汰的是脏页就先写回（`disk_writes` 加 1）；
- `fsync()`：把所有脏页写回（每页 `disk_writes` 加 1），然后它们变成干净页，仍留在缓存里；
- `stats()`：返回 `{"hits": ..., "disk_reads": ..., "disk_writes": ..., "dirty": 当前脏页数}`。

```python
c = PageCache(2)
c.write(1); c.write(1); c.write(2)      # 同一页写两次，只是同一个脏页
c.fsync()                               # 写回 2 页
c.read(3)                               # 读盘，淘汰最久没用的第 1 页（已经是干净页，不用写）
c.stats()    # {"hits": 0, "disk_reads": 1, "disk_writes": 2, "dirty": 0}
```

<!-- 题解 -->
这就是 `write` 快、`fsync` 慢的原因：`write` 只改内存里的页、打个"脏"标记，同一页反复写只算一个脏页（写合并）；真正的磁盘写发生在 `fsync` 或者淘汰的时候。淘汰一个脏页要先写回，所以脏页太多、内存又紧时，读操作都可能被拖慢——内核用 `dirty_background_ratio` 让后台线程提前写回，用 `dirty_ratio` 在脏页过多时让写入者自己参与写回。

实现上用 `OrderedDict` 保存 `页 → 是否脏`，读写都 `move_to_end`，淘汰用 `popitem(last=False)`。`fsync` 只写脏页：把干净页也写一遍既浪费又会让统计失真。
