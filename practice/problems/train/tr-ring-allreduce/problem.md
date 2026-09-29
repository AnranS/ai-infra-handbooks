---
title: 模拟 ring all-reduce
chapter: basics/collectives.md
difficulty: 中等
tags: [集合通信, all-reduce, ring, reduce-scatter, all-gather]
---
`n` 个 rank 排成一个环，每个 rank 有一个长度为 `L` 的列表（`L` 能被 `n` 整除），按顺序切成 `n` 块，每块 `L / n` 个元素。ring all-reduce 分两个阶段：

1. **reduce-scatter**，共 `n - 1` 步。第 `k` 步（`k` 从 0 开始），rank `r` 把自己的第 `(r - k) mod n` 块发给 rank `(r + 1) mod n`，接收方把收到的块**逐元素加**到自己的同一块上。`n - 1` 步之后，rank `r` 的第 `(r + 1) mod n` 块就是所有 rank 这一块的总和；
2. **all-gather**，再 `n - 1` 步。第 `k` 步，rank `r` 把自己的第 `(r + 1 - k) mod n` 块发给下一个 rank，接收方直接**覆盖**自己的这一块。

实现 `ring_allreduce(data)`：`data` 是 `n` 个列表（不要修改它），返回 `(result, sent)`：

- `result`：`n` 个列表，每个都等于所有输入的逐元素总和；
- `sent`：长度为 `n` 的列表，`sent[r]` 是 rank `r` 一共发送的元素个数。

注意每一步里所有 rank 是**同时**发送的：先取出这一步每个 rank 要发的块，再统一加到接收方上；否则同一步里后处理的 rank 会发出已经被改过的数据。

```python
ring_allreduce([[1, 2], [3, 4]])      # ([[4, 6], [4, 6]], [2, 2])
```

<!-- 题解 -->
每个 rank 两个阶段各发 `n - 1` 块，共发送 $2\frac{n-1}{n}L$ 个元素：卡数再多也不超过 $2L$，这就是 ring 算法"带宽最优"的含义。
代价是步数 $2(n-1)$ 随卡数线性增长，每一步都有一次固定的延迟；所以消息小、卡数多时，NCCL 会改用 tree 算法或节点内外分层的算法。
DDP 的梯度同步就是一次 all-reduce；ZeRO 把它拆成了这里的两个阶段：reduce-scatter 之后每个 rank 只更新自己那一块，再 all-gather 参数。
