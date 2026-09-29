---
title: 估算：专家并行的 all-to-all 通信量与节点限制路由
chapter: distributed/expert-parallel.md
difficulty: 中等
tags: [估算, 专家并行, all-to-all, MoE]
---
专家并行（EP）把专家分到 `ep` 张卡上。每个 token 要把隐藏向量发给它选中的 `top_k` 个专家（dispatch），算完再把结果收回来（combine）。
专家均匀分布时，一个被选中的专家落在别的卡上的概率是 $(ep-1)/ep$。约定 dispatch 用 FP8（每元素 1 字节，忽略缩放因子），combine 用 BF16（2 字节）。

跨机时更贵：如果每个 token 发往一个节点只发一份、到了节点内再用 NVLink 转发，跨机流量就只和"token 选中的专家分布在几个节点上"有关。
DeepSeek-V3 这样的模型会**限制每个 token 最多路由到 M 个节点**来压跨机流量。

实现：

1. `a2a_bytes(tokens_per_gpu, hidden, top_k, ep, dispatch_bytes=1, combine_bytes=2)`：返回 `(dispatch, combine)`，每张卡每步发出的字节数（按期望算）；
2. `expected_nodes(top_k, n_nodes)`：一个 token 的 `top_k` 个专家**独立、均匀**地落在 `n_nodes` 个节点上时，涉及的不同节点数的期望 $n\,(1 - (1-1/n)^k)$；
3. `inter_node_bytes_per_token(hidden, top_k, n_nodes, max_nodes=None, dispatch_bytes=1)`：dispatch 时一个 token 的跨机字节数：
   涉及的节点数（有 `max_nodes` 时取两者较小值）× `hidden` × `dispatch_bytes`，其中本节点不算跨机，按 $(n-1)/n$ 折算。

```python
a2a_bytes(4096, 7168, 8, 64)                      # 每卡每步 dispatch 约 231 MB、combine 约 462 MB
expected_nodes(8, 8)                               # 约 5.25 个节点
inter_node_bytes_per_token(7168, 8, 8, max_nodes=4)   # 限制到 4 个节点：跨机流量降到约 76%
```

<!-- 题解 -->
- EP 的通信量和 `top_k × hidden` 成正比、和 EP 规模几乎无关（只差 $(ep-1)/ep$），所以 EP 可以做得很大；它的难点在于 all-to-all 的小包、负载不均和跨机带宽；
- 节点限制路由把"平均 5.25 个节点"压到"最多 4 个"，跨机流量降低约 24%，代价是路由稍微偏离纯 top-k，需要训练时就这样约束；
- 实际系统（例如 DeepEP）把跨机 RDMA 和机内 NVLink 转发分开做，decode 时还会用纯 RDMA 的低延迟模式，把通信和计算用双 micro-batch 重叠起来。
