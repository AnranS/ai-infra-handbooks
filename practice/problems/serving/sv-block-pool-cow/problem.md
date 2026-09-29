---
title: 块池、fork 与写时复制
chapter: engine/paged-kv.md
difficulty: 中等
tags: [PagedAttention, 引用计数, 写时复制]
---
并行采样（`n > 1`）和束搜索时，多个序列共享同一段提示词的 KV：vLLM 让它们的块表指向同一批物理块，用引用计数管理；
某个序列要往一个**被共享的、没写满**的块里追加 token 时，先把这个块复制一份（copy-on-write）。实现：

**`BlockPool(num_blocks)`**（与正文一致）：`allocate(n)` 从空闲队列头部取 `n` 个块、引用计数置 1，不够时返回 `None`；`free(blocks)` 引用计数减 1，减到 0 的块放回空闲队列尾部；`num_free()`；
另外增加 `share(blocks)`：这些块的引用计数加 1；`ref_cnt` 是引用计数列表。

**`Sequence(pool, block_size)`**：

- `num_tokens`：当前 token 数；`block_table`：块表（列表）；
- `append(n)`：追加 `n` 个 token，需要时分配新块。如果最后一个块被共享（引用计数 > 1）并且没写满，先做写时复制：分配一个新块替换块表里的最后一项，旧块引用计数减 1，并在 `pool.copies` 列表里记下 `(旧块, 新块)`（实际引擎会据此复制 KV）。空间不够时抛出 `MemoryError`，并且**不改变任何状态**；
- `fork()`：返回一个新序列，token 数相同，块表是**同一批块**（引用计数加 1）；
- `slot_mapping(positions)`：`[block_table[p // bs] * bs + p % bs for p in positions]`；
- `free()`：释放所有块，`num_tokens` 归零。

<!-- 题解 -->
`append` 的步骤：

1. 若 `num_tokens % bs != 0`（最后一块没写满）且 `ref_cnt[last] > 1`：需要 1 个块做复制；
2. 追加后需要的块数 `ceil((num_tokens + n) / bs)`，减去现有块数，是新分配的块数；
3. 先检查 `num_free() >= 复制所需 + 新块数`，不够就抛 `MemoryError`（先检查、后修改，保证失败时状态不变）；
4. 执行复制和分配。

最后一块正好写满时不需要复制：新 token 会写进新块，共享的块不会被修改。
