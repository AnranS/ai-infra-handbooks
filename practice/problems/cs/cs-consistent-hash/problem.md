---
title: 一致性哈希环与有界负载
chapter: dist/hash-shard.md
difficulty: 中等
tags: [一致性哈希, 虚拟节点, 有界负载, 路由]
---
实现一个带虚拟节点、支持有界负载的一致性哈希环。哈希函数已经给好（`h(s)` 返回一个 64 位整数），必须用它，否则结果对不上。

```python
def h(s):
    import hashlib
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
```

- `Ring(nodes, vnodes=100)`：每个节点在环上放 `vnodes` 个虚拟节点，第 `v` 个点的位置是 `h(f"{node}#{v}")`；
- `route(key)`：返回 `h(key)` **顺时针**方向遇到的第一个虚拟节点所属的节点（超过最大点则回到环的开头）；
- `route_bounded(key, load, cap)`：从同一位置开始顺时针扫，跳过 `load[node] >= cap` 的节点，返回第一个没超上限的；全都超了就返回 `route(key)` 的结果。同一个节点的多个虚拟节点要跳过重复判断（不影响结果，但别重复计数）；
- `rebalance_ratio(other, keys)`：这个环和 `other` 相比，`keys` 里有多少比例的键归属变了（返回 0～1 的浮点数）。

```python
r8 = Ring([f"node{i}" for i in range(8)])
r9 = Ring([f"node{i}" for i in range(9)])
keys = [f"k{i}" for i in range(10000)]
round(r8.rebalance_ratio(r9, keys), 2)      # 约 0.11：加第 9 台只迁移 1/9
```

<!-- 题解 -->
环用一个排好序的 `(位置, 节点)` 列表表示，`route` 就是 `bisect` 加取模回绕。虚拟节点越多分布越均匀：每台一个点时最重的节点能到平均的 2 倍以上，100 个点时降到 1.2 倍左右。

`route_bounded` 是"有界负载的一致性哈希"：热键会让它归属的节点过载，于是顺时针顺延到下一个没超上限的节点。上限越接近平均负载，越均衡，但越多的键被改派、亲和性（缓存命中）越差。注意扫描时要按**节点**去重，一个节点的 100 个虚拟节点在环上是分散的，扫到它的第二个点时不该再判断一次。

`rebalance_ratio` 就是一致性哈希存在的理由：取模分片加一台机器要迁移近 90% 的键，一致性哈希只要 1/(n+1)。
