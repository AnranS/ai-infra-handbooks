---
title: 注意力后端的元数据
chapter: compute/attention.md
difficulty: 中等
tags: [注意力, 变长, 因果掩码]
---
mini-sglang 的注意力后端在每个 batch 开始前调用一次 `prepare_metadata`，把请求信息整理成 kernel 需要的格式（每层共用）。实现参考后端的元数据和前向（numpy 版）：

**`prepare_metadata(reqs, padded_reqs, page_table)`**：`reqs`、`padded_reqs` 是 `(table_idx, cached_len, device_len)` 的列表（`padded_reqs` 是 CUDA Graph 补齐后的，可能多出 dummy 请求；元数据要为 `padded_reqs` 生成）。返回字典：

| 键 | 含义 |
| --- | --- |
| `cu_seqlens_q` | `[0, e0, e0+e1, ...]`，`e = device_len - cached_len`（本轮的 query 数） |
| `cache_seqlens` | 每个请求的 `device_len`（KV 总长） |
| `max_seqlen_k` | `cache_seqlens` 的最大值 |
| `page_table` | `page_table[table_idx, :max_seqlen_k]` 按请求堆叠成 `(bs, max_seqlen_k)` |
| `last_indices` | 只对真实请求（`reqs`）：每个请求最后一个 query 在展平后的 q 里的下标（LM head 只算这些位置） |

**`forward(q, k_pool, v_pool, meta)`**：`q` 形状 `(N, H, d)`（N 是本轮所有 query，已写好 KV）；`k_pool`、`v_pool` 形状 `(num_slots, H_kv, d)`。
对每个请求：从 `meta["page_table"]` 取出前 `cache_seqlens[i]` 个位置的 KV，query 是序列的最后 `e` 个位置，右下角对齐的因果掩码；GQA；缩放 $1/\sqrt{d}$。返回 `(N, H, d)`。

<!-- 题解 -->
与书中 `TorchAttnBackend.prepare_metadata` / `forward` 一致。注意三点：

- 元数据用 `padded_reqs`：dummy 请求也要有合法的元数据（它的 KV 写进专门的 dummy 页），否则 CUDA Graph replay 时会读到垃圾；
- `last_indices = cu_seqlens_q[1:len(reqs)+1] - 1`，只取真实请求；
- page table 按 `max_seqlen_k` 截取，短请求后面是无关的值，用 `cache_seqlens` 截断就不会读到。
