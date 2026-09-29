---
title: 缓存感知的请求路由
chapter: engine/prefix-cache.md
difficulty: 中等
tags: [路由, 前缀缓存, 负载均衡]
---
多个推理副本时，把请求发到"已经缓存了它的前缀"的副本上，可以省掉大量 prefill（SGLang Router 的 cache-aware 策略）。
但只看缓存会把相似的请求全挤到一个副本上，所以要和负载一起权衡。实现 `CacheAwareRouter(num_replicas, block_size, capacity_blocks, load_weight)`：

- 路由器为每个副本维护一个**近似的缓存索引**：一组块哈希（链式哈希，只看装满的块，同前缀缓存那道题：`h_0 = hash((None, 块0))`，`h_i = hash((h_{i-1}, 块i))`）。每个副本最多记 `capacity_blocks` 个哈希，超出时按 LRU 淘汰；
- `route(token_ids)`：对每个副本计算**命中块数** `m`（从头开始连续命中的块数）和当前负载 `load`（正在处理的请求数），
  打分 `score = m - load_weight * load`，选分数最高的副本；分数相同选负载小的；再相同选编号小的。然后：
  - 把这个请求所有装满块的哈希记进被选副本的索引（命中的和新的都算一次"访问"，更新 LRU 顺序，按块的顺序依次访问）；
  - 被选副本的负载加 1；返回 `(副本编号, 命中块数)`；
- `finish(replica)`：该副本负载减 1。

<!-- 题解 -->
每个副本的索引用 `OrderedDict`：访问时 `move_to_end`，插入后超出容量 `popitem(last=False)`。命中块数就是按顺序查哈希、遇到第一个不在索引里的就停。

`load_weight` 决定取舍：0 表示只看缓存（容易热点），很大时退化成"最少连接"（缓存命中率低）。
路由器看不到副本真实的缓存状态（副本会自己淘汰），所以它维护的是一个近似索引——SGLang Router 用的是基数树，思路一样。
