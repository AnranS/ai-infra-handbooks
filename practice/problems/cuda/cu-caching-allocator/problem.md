---
title: 显存缓存分配器：分池、最佳适配、切分与合并
chapter: framework/cuda-runtime.md
difficulty: 困难
tags: [缓存分配器, 显存, 碎片, 内存池]
---
按 PyTorch `CUDACachingAllocator` 的规则（简化版）实现 `CachingAllocator(capacity)`，`capacity` 是驱动最多能给的字节数：

- **取整**：请求的大小向上取整到 512 的倍数（至少 512）；
- **分池**：取整后不超过 1 MiB 的走小池，否则走大池。两个池、不同 stream 的空闲块互不借用；
- **找空闲块**：在同一个池、同一个 stream 的空闲块里找**最佳适配**——大小不小于请求的块里最小的那个，一样大时取地址小的；
- **向驱动要新段**（没找到时）：小池每次 2 MiB；大池请求小于 10 MiB 时要 20 MiB，否则按 2 MiB 向上取整。新段的起始地址从 0 开始依次往后排（用过的地址不再复用）。要了之后总量会超过 `capacity` 时，先把**完全空闲**的段还给驱动（`empty_cache`）再试，还不够就抛出 `MemoryError`；
- **切分**：选中的块比请求大时，剩余部分在小池里不少于 512 字节、在大池里大于 1 MiB 才切下来（成为地址更高的一个空闲块），否则整块都给这个请求（统计里按整块算）；
- **释放**：块变成空闲，并和同一段里前后相邻的空闲块**合并**。

接口：`malloc(size, stream=0)` 返回一个有 `addr`、`size` 属性的块；`free(block)`（释放一个不是已分配状态的块时抛出 `ValueError`）；`empty_cache()`；`memory_allocated()`（已分配块的大小之和）；`memory_reserved()`（所有段的大小之和）；`snapshot()` 返回按地址排序的段列表，每段是 `(base, size, stream, blocks)`，`blocks` 是这一段里按地址排序的 `(addr, size, allocated)`。

```python
MB = 1 << 20
al = CachingAllocator(64 * MB)
a = al.malloc(1000)                 # 取整到 1024，小池新段 [0, 2 MiB)，a.addr == 0
b = al.malloc(5 * MB)               # 大池新段 [2 MiB, 22 MiB)，切出 5 MiB
al.memory_allocated(), al.memory_reserved()      # (1024 + 5 MiB, 22 MiB)
```

<!-- 题解 -->
几个数字都来自 PyTorch 的实现：`kMinBlockSize = 512`、`kSmallSize = 1 MiB`、`kSmallBuffer = 2 MiB`、`kLargeBuffer = 20 MiB`、`kMinLargeAlloc = 10 MiB`、`kRoundLarge = 2 MiB`。小请求多、大小各异，放在 2 MiB 的小段里切；大请求单独成段，避免把大段切得七零八落。大池只在剩余超过 1 MiB 时才切，是为了不留下太多无法再用的小碎片。

测试里有一个经典的"碎片 OOM"：`reserved - allocated` 还有 24 MiB 空闲，却分配不出 12 MiB——空闲块分散在两个段里，每块都不够大，段之间也不能合并。这就是 OOM 报错信息里"reserved but unallocated"很大的原因。缓解办法：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（段可以用虚拟内存映射往后扩展，相邻的空闲空间就连起来了）、`max_split_size_mb`（大块不再切分），或者像推理引擎那样启动时就把 KV Cache 一次分配好。

用的数据结构：每段是一条按地址排好的双向链表，合并只看前后邻居；空闲块按 `(stream, 大小, 地址)` 放进有序集合，最佳适配就是一次二分查找（这里用线性扫描也可以）。
