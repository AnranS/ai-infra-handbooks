---
title: 多 block 前缀和
chapter: kernels/scan.md
difficulty: 困难
tags: [gpusim, scan, 前缀和]
---
实现任意长度数组的**包含式前缀和**（inclusive scan）：`out[i] = x[0] + ... + x[i]`（int32）。数组比一个 block 大，需要三步：

1. `scan_blocks(t, x, out, block_sums, n)`（`block = 64`）：每个 block 对自己的 64 个元素做 block 内的前缀和，写到 `out`，并把这一段的总和写进 `block_sums[blockIdx.x]`；
2. 对 `block_sums` 递归地做前缀和（它可能也超过 64 个元素）；
3. `add_offsets(t, out, scanned_sums, n)`：第 `b` 个 block（`b >= 1`）的每个元素加上 `scanned_sums[b - 1]`。

主机函数 `inclusive_scan(x)`：输入 int32 的 numpy 数组，返回同样长度的前缀和（numpy 数组）。

（为了让模拟器跑得快，这里 block 只有 64 个线程；真实 GPU 上常用 256～1024。）

block 内的前缀和用 **Hillis–Steele** 算法：`offset = 1, 2, 4, ..., 32`，每轮 `tmp = s[i - offset] + s[i]`（`i >= offset` 时）。
注意：同一轮里有的线程在读 `s[i - offset]`，有的线程在写 `s[i]`——直接原地更新会产生数据竞争（gpusim 会报出来）。
要么用两块缓冲区交替读写（double buffering），要么每轮"先读到寄存器、屏障、再写"。

<!-- 题解 -->
block 内（先读后写的写法）：

```python
s[tid] = x[i] if i < n else 0
yield t.syncthreads()
offset = 1
while offset < 64:
    v = s[tid - offset] if tid >= offset else 0
    yield t.syncthreads()          # 所有人读完再写
    s[tid] += v
    yield t.syncthreads()
    offset *= 2
```

主机端递归：`sums = scan(block_sums)`，然后 `add_offsets`。总工作量 $O(n \log B)$（$B$ 是 block 大小），Blelloch 算法可以做到 $O(n)$，
而工业界的实现（CUB 的 decoupled look-back）只需要一遍全局内存读写。
