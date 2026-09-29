---
title: 共享内存矩阵转置
chapter: kernels/transpose.md
difficulty: 中等
tags: [共享内存, 合并访存, bank conflict]
---
用 [gpusim](#/local) 写一个矩阵转置 kernel：输入 `a` 形状 `(rows, cols)`，输出 `b` 形状 `(cols, rows)`，`b[j, i] = a[i, j]`。

启动配置固定为 `block = (32, 8)`、`grid = (cdiv(cols, 32), cdiv(rows, 32))`：每个 block 负责一个 32×32 的块，每个线程搬 4 个元素。
你需要实现 `transpose_kernel(t, a, b, rows, cols)`，要求：

1. **结果正确**，包括行数、列数不是 32 倍数的情况；
2. **读写都合并**：全局内存的读、写效率都要达到 100%（`stats.load_efficiency` 和 `stats.store_efficiency`）；
3. **没有 bank conflict**（`stats.bank_conflicts == 0`）。

直接写 `b[x, y] = a[y, x]` 时，读是合并的，写却是跨步的（相邻线程写的地址相差 `rows` 个元素）。
标准做法是先把块读进共享内存，`__syncthreads()` 之后换一种方式读出来再写回。

gpusim 的写法（和 CUDA 一一对应）：

```python
tile = t.shared("tile", (32, 33))          # __shared__ float tile[32][33];
tile[ty, tx] = a[y, x]                     # 按元素访问，二维下标用逗号
yield t.syncthreads()                      # __syncthreads();
```

<!-- 题解 -->
读阶段：线程 `(tx, ty)` 读 `a[by*32 + ty + j, bx*32 + tx]`（同一 warp 的 32 个线程 `tx` 连续，读连续地址），写进 `tile[ty + j, tx]`。

写阶段：交换块的坐标，线程读 `tile[tx, ty + j]`，写 `b[bx*32 + ty + j, by*32 + tx]`，写的地址又是连续的。

bank conflict 出在读 `tile[tx, ty + j]`：一个 warp 的 32 个线程访问同一列，如果每行 32 个 float，这 32 个地址落在同一个 bank 上。
把每行补一个元素（`(32, 33)`），同一列的相邻元素就错开一个 bank，冲突消失。

边界：读和写都要各自判断坐标是否越界；越界的线程不读不写，但**仍然要执行 `yield t.syncthreads()`**。
