---
title: 估算：MoE 的激活参数与 decode 时要读多少专家
chapter: transformer/moe.md
difficulty: 中等
tags: [估算, MoE, 参数量, 带宽]
---
MoE 模型"总参数 671B、激活 37B"是怎么算出来的？decode 时一个 batch 要从显存读多少专家权重？`cfg` 的字段：

- `d`：隐藏维度；`n_layers`：总层数，其中前 `n_dense_layers` 层是普通 FFN（中间维度 `dense_ffn`），其余是 MoE 层；
- MoE 层有 `n_experts` 个路由专家、`n_shared` 个共享专家，每个 token 选 `top_k` 个路由专家；每个专家是 SwiGLU FFN，中间维度 `moe_ffn`，参数 $3 \cdot d \cdot \text{moe\_ffn}$；
- 每个 MoE 层还有一个路由器（$d \times \text{n\_experts}$）；每层注意力的参数量直接给出：`attn_params`；
- 词表 `vocab`，输入嵌入和输出层不共享（$2 \cdot \text{vocab} \cdot d$）；忽略归一化层的参数。

实现：

1. `moe_params(cfg)`：返回 `(total, active)`，激活参数是每个 token 实际用到的参数（只算 `top_k` 个路由专家）；
2. `expected_distinct_experts(n_experts, top_k, n_tokens)`：一个 MoE 层里，`n_tokens` 个 token 各自**均匀、独立**地选 `top_k` 个专家时，被选中的不同专家数的期望；
3. `decode_expert_bytes(cfg, batch, bytes_per_param=1)`：decode 一步（batch 个 token）所有 MoE 层要读的专家权重字节数（被选中的路由专家 + 共享专家，按期望算）。

```python
moe_params(dsv3)                          # 约 (671e9, 37.5e9)
expected_distinct_experts(256, 8, 1)      # 8.0
expected_distinct_experts(256, 8, 64)     # 约 222：batch 64 时几乎所有专家都被读到
```

<!-- 题解 -->
一个专家不被某个 token 选中的概率是 $1 - k/E$，$T$ 个 token 都不选它的概率是 $(1-k/E)^T$，所以期望的不同专家数是 $E\,(1 - (1-k/E)^T)$。

这解释了 MoE decode 的一个关键事实：batch 稍大，几乎每个专家都会被读到，**每步要读的权重接近总参数量而不是激活参数量**。
所以大 MoE 在线服务要用专家并行把专家分散到很多卡上，让每张卡只读自己那部分专家；而 batch=1 时只需读激活的那部分，单请求延迟很低。
