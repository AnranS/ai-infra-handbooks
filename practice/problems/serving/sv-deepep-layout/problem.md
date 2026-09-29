---
title: EP 通信的账本：dispatch 布局、接收计划与固定槽位
chapter: comm/nvshmem-deepep.md
difficulty: 中等
tags: [专家并行, DeepEP, all-to-all, MoE]
requires: [numpy]
---
`num_experts` 个专家平均放在 `num_ranks` 张卡上，专家 `e` 在第 `e // (num_experts // num_ranks)` 张卡上，是那张卡的第 `e % (num_experts // num_ranks)` 个本地专家。`topk_idx` 是 `(T, k)` 的整数数组，每个 token 选中的专家，`-1` 表示这个位置没有专家（被掩掉了）。

1. `dispatch_layout(topk_idx, num_experts, num_ranks)`：返回 `(tokens_per_rank, tokens_per_expert, in_rank)`。`in_rank[t, r]` 表示 token `t` 至少有一个专家在第 `r` 张卡上；`tokens_per_rank[r]` 是要发往第 `r` 张卡的 token 数（**同一个 token 去同一张卡只发一份**）；`tokens_per_expert[e]` 是选中专家 `e` 的次数。这就是 DeepEP `get_dispatch_layout` 算的东西；
2. `recv_plan(all_topk, num_experts)`：高吞吐模式里，每张卡要先知道自己会从每张卡收到多少个 token，才能分配接收缓冲区。`all_topk[s]` 是第 `s` 张卡上的 `topk_idx`（卡数 = `len(all_topk)`）。返回 `(counts, offsets, totals)`：`counts[s][d]` 是从 `s` 发往 `d` 的 token 数（同样按 token 去重），`offsets[s][d]` 是这些 token 在 `d` 的接收缓冲区里从第几行开始放（按来源卡的编号依次排），`totals[d]` 是 `d` 一共收多少行；
3. `ll_slots(topk_idx, src_rank, num_experts, num_ranks, max_tokens)`：低延迟模式不交换数量，每张卡为"每个本地专家 × 每个来源卡"预留 `max_tokens` 个槽位。发送方自己决定每份数据写进哪个槽位：token 按编号从小到大，发往同一个专家的第几份就写第几个槽位。返回 `(T, k)` 的槽位编号（`-1` 的位置也是 `-1`）；某个专家的份数超过 `max_tokens` 时抛出 `OverflowError`；
4. `ll_buffer_bytes(num_experts, num_ranks, max_tokens, msg_bytes)`：低延迟模式每张卡的接收缓冲区字节数。

```python
topk = np.array([[0, 3], [1, 2], [3, -1]])       # 4 个专家、2 张卡：专家 0、1 在卡 0，专家 2、3 在卡 1
dispatch_layout(topk, 4, 2)
# (array([2, 3]), array([1, 1, 1, 2]), array([[True, True], [True, True], [False, True]]))
```

<!-- 题解 -->
去重是按"目标卡"而不是"目标专家"：一个 token 的两个专家在同一张卡上时只发一份隐藏向量，接收方在本地复制给两个专家。`recv_plan` 就是高吞吐模式里那次"交换每个目标收多少个 token"：数量在 GPU 上算出来，接收方要拷回 CPU 才能分配缓冲区——这次同步让它不能录进 CUDA Graph。

低延迟模式用固定槽位换掉了这次同步：接收缓冲区按最坏情况预留（`本地专家数 × 卡数 × max_tokens × 每条消息的字节数`），发送方自己算出槽位就直接写（用 NVSHMEM 或 IBGDA 远程写），接收方不需要事先知道数量，形状全部固定。按本章的规模（256 个专家、64 张卡、每卡最多 128 个 token、每条 7408 字节）是 232 MiB，真正用到的只有百分之几。
