---
title: warp shuffle 归约
chapter: basics/sync-warp.md
difficulty: 中等
tags: [gpusim, warp shuffle, 归约]
---
用 warp 级原语实现求和，不经过共享内存交换每个元素。

1. 设备函数（生成器）`warp_sum(t, v)`：用 `__shfl_down_sync` 的思路在一个 warp 内求和，返回后 **lane 0** 持有 32 个值的和。
   在 gpusim 里调用 warp 操作要写 `v2 = yield t.shfl_down(v, offset)`，所以 `warp_sum` 本身也是生成器，调用它要写 `s = yield from warp_sum(t, v)`；
2. kernel `block_sum(t, x, out, n)`：`block = 256`，每个线程用 grid-stride 循环累加若干个元素，然后：
   - 每个 warp 用 `warp_sum` 归约；
   - 各 warp 的 lane 0 把结果写进共享内存 `partial`（**最多 32 个 float**）；
   - `__syncthreads()` 之后，第 0 个 warp 读出 `partial` 再做一次 `warp_sum`；
   - 线程 0 用 `atomicAdd` 把 block 的和加到 `out[0]`。

测试检查：结果正确；共享内存每个 block 只用了不超过 32 个 float；每个 block 只做 1 次 `__syncthreads()`；每个 block 恰好 1 次原子操作。

<!-- 题解 -->
```python
def warp_sum(t, v):
    offset = 16
    while offset > 0:
        v += yield t.shfl_down(v, offset)
        offset //= 2
    return v
```

`shfl_down(v, offset)`：lane $i$ 拿到 lane $i + \text{offset}$ 的值（超出 warp 的拿到自己的值），5 轮之后 lane 0 就是总和。
第二级：`partial[warp] = s`（lane 0 写），屏障后第 0 个 warp 的 lane $i$ 读 `partial[i]`（$i$ 超过 warp 数时读 0），再归约一次。
**所有**线程都要执行 `yield t.syncthreads()`，第 0 个 warp 的 32 个线程都要参与第二次 shuffle（不能只让 lane 0 进去）。
