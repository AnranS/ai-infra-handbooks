---
title: PD 分离：跨实例搬运 KV 的拷贝计划
chapter: distributed/pd-disagg.md
difficulty: 中等
tags: [PD 分离, KV 传输, 合并区间]
---
prefill 实例算完提示词的 KV 后，要把它传给 decode 实例。两边各有自己的分页 KV 池，**块大小可能不同**（例如 prefill 用 16、decode 用 32），块表也完全不同。
传输引擎（NIXL、Mooncake 等）按"连续的一段字节"来搬运，段越少越高效。

一个 token 在池里的线性位置（slot）是 `块号 × block_size + 块内偏移`；把一个池看成一维的 slot 数组。实现 `transfer_plan(num_tokens, src_table, src_bs, dst_table, dst_bs)`：

- 返回拷贝段列表 `[(src_slot, dst_slot, length), ...]`，表示把源池 `[src_slot, src_slot + length)` 的 token 拷到目标池 `[dst_slot, dst_slot + length)`；
- 覆盖该请求的全部 `num_tokens` 个 token，每个 token 恰好拷贝一次，按 token 顺序排列；
- **段数最少**：只要相邻 token 在源池和目标池里都连续（两边的 slot 都恰好加 1），就要合并进同一段。

再实现 `apply_plan(src_pool, dst_pool, plan)`：按计划拷贝（`src_pool`、`dst_pool` 是形状 `(num_slots, ...)` 的 numpy 数组）。

<!-- 题解 -->
逐 token 求源和目标的 slot：`s = src_table[p // src_bs] * src_bs + p % src_bs`，目标同理；
如果 `s == 上一段的源结尾` 且 `d == 上一段的目标结尾`，就延长上一段，否则开新段。复杂度 $O(\text{num\_tokens})$；
也可以按"两边块边界的并集"切分，复杂度降到块数级别。

块表里恰好连续的块号（比如 7、8、9）会让段变长。PD 分离的调度器常常为此在 decode 侧优先分配连续的块。
