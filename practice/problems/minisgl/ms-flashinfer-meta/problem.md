---
title: FlashInfer 的分页元数据
chapter: perf/gpu-attention.md
difficulty: 中等
tags: [FlashInfer, CSR, 分页注意力]
---
FlashInfer 的分页注意力接口用 **CSR 格式**描述每个请求的 KV 页：

| 数组 | 含义 |
| --- | --- |
| `qo_indptr` | 长度 B+1，第 i 个请求的 query 是 `q[qo_indptr[i]:qo_indptr[i+1]]` |
| `kv_indptr` | 长度 B+1，第 i 个请求的页是 `kv_indices[kv_indptr[i]:kv_indptr[i+1]]` |
| `kv_indices` | 所有请求的页号首尾相接 |
| `kv_last_page_len` | 长度 B，每个请求最后一页里有几个有效 token（1～page_size） |

mini-sglang 的 page table 是**按 token** 存的（`page_table[row, j]` 是第 j 个 token 的位置），`page_size = 1` 时页号就是 token 位置、`last_page_len` 恒为 1（书中就是这样用 FlashInfer 的）。
这道题实现一般的 `page_size`：

`flashinfer_metadata(reqs, page_table, page_size)`：`reqs` 是 `[(row, extend_len, seq_len), ...]`（本轮新算的 token 数、算完后的 KV 长度）。第 j 页的页号是 `page_table[row, j * page_size] // page_size`。返回字典（numpy int32 数组）：`qo_indptr`、`kv_indptr`、`kv_indices`、`kv_last_page_len`。

再实现 `paged_attention_from_meta(q, k_cache, v_cache, meta, page_size)`，只用这些元数据做一次注意力（验证元数据正确）：
`k_cache`、`v_cache` 形状 `(num_pages, page_size, H, d)`；第 i 个请求的 KV 是它的页展开后的前 `seq_len` 个 token（`seq_len = (页数 - 1) * page_size + last_page_len`）；query 是序列的最后 `extend_len` 个位置，按右下角对齐的因果掩码。返回 `(N, H, d)`。

<!-- 题解 -->
```python
pages_i = ceil(seq_len / ps)
kv_indices += [page_table[row, j * ps] // ps for j in range(pages_i)]
last_page_len = seq_len - (pages_i - 1) * ps          # 即 (seq_len - 1) % ps + 1
```

CSR 格式让一个 kernel 处理任意数量、任意长度的请求：每个线程块根据 `indptr` 找到自己负责的请求和页。
FlashInfer 的 `plan` 阶段在 CPU 上根据这些元数据做负载均衡（把长序列拆成多段分给多个线程块），`run` 阶段只做计算——这就是书中说的 plan/run 分离。
