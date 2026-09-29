---
title: 张量并行的权重切分（含 KV 头复制）
chapter: perf/tensor-parallel.md
difficulty: 中等
tags: [张量并行, GQA, 词表并行]
---
实现书中 `shard_tensor(key, value, rank, size, num_kv_heads)`：从完整的权重里取出第 `rank` 个分片（权重是 PyTorch 的布局 `(out, in)`，这里用 numpy 数组）。规则：

| 名字里含 | 切法 |
| --- | --- |
| `.q_proj`、`.k_proj`、`.v_proj`、`.gate_proj`、`.up_proj` | 列并行：沿第 0 维均分成 `size` 份 |
| `.o_proj`、`.down_proj` | 行并行：沿第 1 维均分 |
| `lm_head`、`embed_tokens` | 词表并行：沿第 0 维，每份 `ceil(V / size)` 行，**最后一份可能短一些** |
| 其他（norm、MoE 路由等） | 每个 rank 一份完整的 |

特殊情况：**KV 头比 rank 少**（`.k_proj` / `.v_proj` 且 `num_kv_heads < size`）时，几个 rank 共用同一个 KV 头，各自保存一份副本：
`head_dim = value.shape[0] // num_kv_heads`，第 `rank` 个 rank 用第 `rank * num_kv_heads // size` 个头。

`size == 1` 时原样返回。返回的数组要是**副本**（不能和原数组共享内存，否则释放完整张量时省不下内存）。

再实现 `div_even(a, b, allow_replicate=False)`：`a // b`，要求整除（否则 `ValueError`）；`allow_replicate=True` 且 `b > a` 时，要求 `b % a == 0` 并返回 1。
模型里每个 rank 的 KV 头数就是 `div_even(num_kv_heads, size, allow_replicate=True)`。

<!-- 题解 -->
均分用 `np.split(value, size, axis=d)[rank]`（要求整除；`torch.chunk` 在不整除时行为不同，书中的维度都能整除）。

Qwen3-0.6B 有 8 个 KV 头，TP=16 时每两个 rank 共用一个 KV 头：rank 0、1 用头 0，rank 2、3 用头 1……
这样注意力仍然不需要通信（每个 rank 的 q 头都能在本地找到对应的 KV 头），代价是 KV 权重和 KV Cache 各多存一份。
