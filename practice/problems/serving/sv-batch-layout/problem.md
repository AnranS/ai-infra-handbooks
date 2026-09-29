---
title: 一次混合前向的批次布局
chapter: engine/batch-layout.md
difficulty: 中等
tags: [变长批处理, slot mapping, 注意力元数据]
---
把若干请求本步要算的新 token 拼成一维（不填充），prefill、decode、分块 prefill 可以混在同一批里。实现两个函数：

**`build_batch(items, block_size)`**：`items` 是 `[(token_ids, num_computed, block_table, need_logits), ...]`，返回一个字典（字段与正文 `BatchInput` 相同，都用 numpy 数组或列表）：

| 键 | 含义 |
| --- | --- |
| `input_ids`、`positions`、`slot_mapping` | 形状 `(N,)`；位置从 `num_computed` 开始 |
| `query_start_loc` | `[0, n0, n0+n1, ...]`（长度 B+1）|
| `seq_lens` | 本步算完后各请求的上下文长度 |
| `logits_indices` | `need_logits` 为真的请求的最后一个新 token 在 N 中的下标 |
| `max_seqlen_q`、`max_seqlen_k` | 各请求新 token 数、上下文长度的最大值（FlashAttention 变长接口要用）|
| `block_tables` | `(B, max_blocks)` 的 int32 数组，每行是一个请求的块表，不足的地方填 `-1` |

**`varlen_paged_attention(q, k_cache, v_cache, batch, scale)`**：`q` 形状 `(N, H, d)`（已经加过 RoPE），`k_cache`、`v_cache` 形状 `(num_blocks, block_size, H_kv, d)`（本步的 K、V **已经写入**）。
对每个请求，取出它的全部 `seq_len` 个 KV，新 token 是序列的最后 `n` 个位置，按"右下角对齐"的因果掩码计算注意力。返回 `(N, H, d)`。GQA：第 `h` 个头用第 `h // (H / H_kv)` 个 KV 头。

<!-- 题解 -->
`build_batch` 照正文写，再补上 `max_seqlen_q = max(qsl[i+1] - qsl[i])`、`max_seqlen_k = max(seq_lens)`，块表用 `np.full((B, max_len), -1)` 再逐行填。

注意力：对第 `i` 个请求，`q_i = q[qsl[i]:qsl[i+1]]`，KV 按块表 gather 出 `(seq_len, H_kv, d)`；
第 `j` 个新 token 的位置是 `seq_len - n + j`，允许的 key 满足 `key_pos <= seq_len - n + j`。
这正是 `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ..., causal=True)` 的语义（它的因果掩码也是右下角对齐）。
