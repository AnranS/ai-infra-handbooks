---
title: 分页 KV Cache 上的 decode 注意力
chapter: advanced/attention.md
difficulty: 中等
tags: [PagedAttention, GQA, block table]
---
vLLM 的 PagedAttention 把每个序列的 KV 存在不连续的物理块里，靠 **block table** 找到它们。用 numpy 实现 decode 阶段（每个序列 1 个 query）的注意力：

```python
paged_decode(q, k_cache, v_cache, block_tables, seq_lens)
```

- `q`：`(B, H_q, d)`，每个序列当前这一步的 query；
- `k_cache`、`v_cache`：`(num_blocks, block_size, H_kv, d)`，所有序列共用的物理块池；
- `block_tables`：`(B, max_blocks)` 的整数数组，第 `b` 个序列的第 `j` 个逻辑块存放在物理块 `block_tables[b, j]`（多余的位置是无效值，可能是 -1）；
- `seq_lens`：`(B,)`，第 `b` 个序列当前有多少个 token 的 KV（包括这一步的，所以 query 能看到全部 `seq_lens[b]` 个 key）；
- 第 `b` 个序列的第 `p` 个 token 在物理块 `block_tables[b, p // block_size]` 的第 `p % block_size` 个槽位；
- GQA：第 `h` 个 query 头用第 `h // (H_q / H_kv)` 个 KV 头；缩放 $1/\sqrt{d}$。

返回 `(B, H_q, d)`。不能访问 `block_tables` 里超出 `ceil(seq_len / block_size)` 的无效项。

<!-- 题解 -->
对每个序列：`nblk = ceil(L / bs)`，`blocks = block_tables[b, :nblk]`，用花式索引一次取出 `k_cache[blocks]`（`(nblk, bs, H_kv, d)`），
reshape 成 `(nblk * bs, H_kv, d)` 再截取前 `L` 个 token。之后就是普通的 GQA 注意力（单个 query，不需要掩码）。

CUDA 版（正文的 `paged_decode.cu`）里一个 thread block 处理一个 (序列, 头)，按块遍历 block table，块内的 token 由不同的 warp 并行处理；
为了让 key 的读取合并，vLLM 的 KV 布局还会把 head_dim 切成若干段交错存放。
