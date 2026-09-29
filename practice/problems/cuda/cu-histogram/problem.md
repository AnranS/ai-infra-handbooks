---
title: 直方图：共享内存私有化
chapter: basics/sync-warp.md
difficulty: 中等
tags: [gpusim, 原子操作, 共享内存]
---
统计 `data`（取值 0～`bins-1` 的整数）里每个值出现的次数。直接对全局内存的 `hist` 做原子加，所有线程抢同一小块地址，冲突很严重。
标准优化是**私有化**：每个 block 先在共享内存里维护一份自己的直方图，最后再合并到全局。

实现 kernel `histogram(t, data, hist, n, bins)`（`block = 128`，grid 大小由测试决定，用 grid-stride 循环）：

1. 用线程协作把共享内存直方图 `local`（`bins` 个 int32，`bins <= 256`）清零，屏障；
2. 每个线程对自己负责的元素做 `t.atomic_add(local, v, 1)`；
3. 屏障后，把 `local` 里**非零**的桶用 `atomic_add` 加到全局 `hist` 上。

测试检查：结果正确；全局内存上的原子操作次数不超过 `grid × bins`。另外故意漏掉第 1 步的屏障会怎样？gpusim 会报告数据竞争，可以试试。

<!-- 题解 -->
```python
local = t.shared("local", 256, dtype=np.int32)
for b in range(t.threadIdx.x, bins, t.blockDim.x):
    local[b] = 0
yield t.syncthreads()
i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
while i < n:
    t.atomic_add(local, data[i], 1)
    i += t.gridDim.x * t.blockDim.x
yield t.syncthreads()
for b in range(t.threadIdx.x, bins, t.blockDim.x):
    c = local[b]
    if c:
        t.atomic_add(hist, b, c)
```

共享内存原子操作比全局的快得多，而且冲突只发生在同一个 block 内。数据分布越集中（比如图像里大片相同颜色），私有化的收益越大。
