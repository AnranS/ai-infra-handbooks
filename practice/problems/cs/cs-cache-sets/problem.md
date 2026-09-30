---
title: 组相联缓存与冲突缺失
chapter: arch/cpu.md
difficulty: 中等
tags: [缓存, 组相联, 冲突缺失, 对齐]
---
缓存不是"哪里有空放哪里"：一个地址先按 `(addr // line) % nsets` 决定落进哪一组，只能在这一组的 `ways` 个位置里挑，组满了按 LRU 淘汰。实现两个函数：

1. `cache_stats(addrs, size, ways, line=64)`：按顺序访问 `addrs` 里的地址，返回 `(hits, misses)`。组数 `nsets = size // (ways * line)`；
2. `pad_floats(rows, cols, size, ways, line=64)`：一个 `rows × cols` 的 float32 矩阵**按列遍历**时，每行末尾要补多少个 float32，才能让这 `rows` 行的起始地址落进 `min(rows, nsets)` 个**不同的组**？补的个数必须是整条缓存行（`line // 4` 的倍数），返回最小的那个（已经够分散就返回 0）。

```python
cache_stats([0, 64, 0, 128], 128, 1)        # (1, 3)：2 组 1 路，地址 0 和 128 争同一组
pad_floats(512, 512, 48 * 1024, 12)         # 16：每行 2048 字节 = 32 条缓存行，512 行只落进 2 个组
```

<!-- 题解 -->
缓存模拟和 TLB 那道题一样用 `OrderedDict` 做每组的 LRU，只是多了一层"先算组号"：`tag = addr // line`、`s = tag % nsets`。注意每一组里要用**完整的行号** `tag` 做键，用组号做键的话，同一组里的不同行会被当成同一行，命中率虚高。

`pad_floats`：第 r 行的起始地址是 `r * (cols + pad) * 4`，它落进的组是 `(addr // line) % nsets`。从 0 开始按整条缓存行递增，数一数这 `rows` 行落进几个不同的组，够 `min(rows, nsets)` 个就停。行距是 `line` 的整数倍时，行与行之间的组号差是固定的步长 `step`，能覆盖的组数是 `nsets / gcd(step, nsets)`，所以 `step` 是 2 的幂的倍数时最糟。

这正是本机 L1（48 KB、12 路、64 组）上 512×512 的 float 矩阵按列遍历几乎全部缺失的原因：行距 2048 字节 = 32 条缓存行，`gcd(32, 64) = 32`，512 行只落进 2 个组，24 个位置装不下。每行补 16 个 float 后步长变成 33，和 64 互质，各行分散到全部 64 个组里。
