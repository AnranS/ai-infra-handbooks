---
title: Ring All-Reduce
chapter: tools/multi-gpu.md
difficulty: 中等
tags: [NCCL, all-reduce, 通信量]
---
NCCL 在多卡间做 all-reduce 的经典算法是 ring：$n$ 张卡连成环，每张卡只和下一张卡通信。实现 `ring_allreduce(data)`：

- `data`：长度为 $n$ 的列表，第 $r$ 个元素是第 $r$ 张卡上的一维 numpy 数组（长度相同）；
- 把每个数组用 `np.array_split(x, n)` 切成 $n$ 块（块 $0 \ldots n-1$）；
- **Reduce-Scatter**（$n-1$ 步）：第 $s$ 步（$s = 0, \ldots, n-2$），每张卡 $r$ 把自己的第 $(r - s) \bmod n$ 块发给卡 $(r+1) \bmod n$，对方把收到的数据**加到**自己的同一块上。所有卡在同一步里是**同时**发送的：发送的都是这一步开始时的值；
- 结束后，卡 $r$ 的第 $(r + 1) \bmod n$ 块是全部卡的和；
- **All-Gather**（$n-1$ 步）：第 $s$ 步，卡 $r$ 把自己的第 $(r + 1 - s) \bmod n$ 块发给下一张卡，对方用它**覆盖**自己的同一块。

返回 `(results, sent)`：`results` 是每张卡最终的数组（都应该等于所有输入之和），`sent` 是每张卡总共发送的元素个数（列表）。不要修改输入数组。

每张卡发送的数据量约为 $2 \cdot \frac{n-1}{n} \cdot N$，和卡数几乎无关——这是 ring 算法能扩展到很多卡的原因。

<!-- 题解 -->
```python
chunks = [np.array_split(x.astype(float64), n) for x in data]   # 深拷贝
for s in range(n - 1):
    msgs = [(r, (r - s) % n, chunks[r][(r - s) % n].copy()) for r in range(n)]   # 先取出这一步要发的
    for r, c, payload in msgs:
        chunks[(r + 1) % n][c] += payload
```

All-Gather 同理，只是把 `+=` 换成赋值。先把本步所有消息取出来再统一应用，就是"同时发送"的语义。
每一步每张卡发送 $N/n$ 个元素，共 $2(n-1)$ 步。总时间约为 $2\frac{n-1}{n}\frac{N}{B} + 2(n-1)\alpha$（$B$ 带宽，$\alpha$ 每步延迟），卡数多时延迟项变大，于是 NCCL 在大规模时改用 tree 算法。
