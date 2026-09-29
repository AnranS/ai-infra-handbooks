---
title: 写 KV Cache 的 kernel
chapter: perf/kernels.md
difficulty: 中等
tags: [gpusim, 合并访存, scatter]
---
每一层注意力算完新 token 的 k、v 之后，要把它们写到 KV 池里 `out_loc` 指定的位置：`k_cache[out_loc[i]] = k[i]`（v 同理）。这是书中自定义 kernel 那一章里的 `store_cache`。
每个 token 的一行有 `row = H × D` 个元素（例如 8 × 128 = 1024 个），这里把它们展平成二维数组：`k` 形状 `(N, row)`，`k_cache` 形状 `(num_slots, row)`。

用 gpusim 实现 kernel `store_cache_kernel` 和主机函数 `store_cache(k, v, k_cache, v_cache, out_loc)`（参数都是设备数组，`out_loc` 是 int32），要求：

- 结果正确；
- 读 `k`、写 `k_cache` 的全局内存效率都 $\ge 90\%$（`stats.array("k")`、`stats.array("k_cache")`）。

模板里是"一个线程搬一个 token"的写法：相邻线程访问的地址相隔整整一行，完全不合并。
提示：每个 block 负责一个 token（`grid = N`），block 里的线程沿着这一行跨步地搬（`for j in range(tid, row, blockDim)`），同一个 warp 访问连续的地址。

<!-- 题解 -->
```python
@gs.kernel
def store_cache_kernel(t, k, v, k_cache, v_cache, out_loc, row):
    i = t.blockIdx.x
    dst = out_loc[i]
    for j in range(t.threadIdx.x, row, t.blockDim.x):
        k_cache[dst, j] = k[i, j]
        v_cache[dst, j] = v[i, j]
```

`out_loc[i]` 被 block 里所有线程读，是一次广播（一个扇区）。书中的 CUDA 版本还会用 `float4` / 128 位向量化，让每个线程一次搬 16 字节，并把 k 和 v 放在同一个 kernel 里减少启动次数。
这个 kernel 是纯粹的带宽瓶颈，合并与否对速度的影响可以有好几倍。
