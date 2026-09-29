---
title: 训练到推理的 qkv 权重重切分
chapter: practice/frameworks-rl.md
difficulty: 中等
tags: [RL, 权重同步, 张量并行, GQA, 重新切分]
---
推理引擎把 Q、K、V 三个投影矩阵融合成一个 `qkv_proj`，再按张量并行切分：rank `r` 的分片依次是"它负责的 Q 头 + 这些 Q 头对应的 K 头 + 对应的 V 头"的行。模型有 `heads` 个 Q 头、`kv_heads` 个 KV 头（GQA，每 `heads / kv_heads` 个 Q 头共用一个 KV 头），每个头 `d` 行。

1. `fused_qkv_rows(rank, tp, heads, kv_heads, d)`：返回 `[("q", 起, 止), ("k", 起, 止), ("v", 起, 止)]`，是这个 rank 的分片在**完整的** Q、K、V 矩阵里对应的行区间（左闭右开）。规则：
    - `heads` 必须能被 `tp` 整除，每个 rank 负责 `heads / tp` 个连续的 Q 头；
    - `kv_heads >= tp` 时要求 `kv_heads` 能被 `tp` 整除，每个 rank 负责 `kv_heads / tp` 个 KV 头；
    - `kv_heads < tp` 时要求 `tp` 能被 `kv_heads` 整除，每个 rank 只放 1 个 KV 头——它的 Q 头所对应的那个（于是每个 KV 头被复制到 `tp / kv_heads` 个 rank 上）；
    - 不满足整除条件时抛出 `ValueError`。
2. `train_sources(rank, infer_tp, train_tp, heads, kv_heads)`：训练端用 Megatron 风格的 TP=`train_tp`（Q、K、V 分开，各自按头连续均分，要求 `train_tp` 能整除 `kv_heads`）。推理端 rank `rank` 的 `qkv_proj` 分片需要从哪些训练 rank 取数据？返回升序的列表。

```python
fused_qkv_rows(1, 4, heads=8, kv_heads=2, d=8)
# [('q', 16, 32), ('k', 0, 8), ('v', 0, 8)]
train_sources(3, infer_tp=4, train_tp=2, heads=8, kv_heads=2)   # [1]
```

<!-- 题解 -->
常见的错误是"先把完整的 Q、K、V 按行拼起来，再均分成 `tp` 份"：那样 rank 0 拿到的全是 Q 头，K、V 都落在最后几个 rank 上。更糟的是，某些配置下形状恰好对得上，前向不会报错，只是结果全错——在 RL 里表现为奖励不涨、rollout 输出乱码。
`train_sources` 说明了另一件事：推理 TP 是训练 TP 的倍数时，每个推理 rank 只需要一个训练 rank 的数据，可以点对点直传，不必先在某处拼出完整权重。
