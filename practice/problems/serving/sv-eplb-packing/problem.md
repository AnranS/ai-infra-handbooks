---
title: EPLB：复制热门专家与均衡打包
chapter: moe/ep-deploy.md
difficulty: 中等
tags: [EPLB, 专家并行, 负载均衡, 贪心]
---
实现开源 EPLB 算法的两个基本步骤（单层）：

1. `replicate(load, n_phys)`：`load[e]` 是逻辑专家 `e` 的负载。一开始每个专家 1 个副本；每次给"单个副本负载 `load[e] / cnt[e]` 最大"的专家加一个副本（并列时取编号小的），直到副本总数为 `n_phys`。返回每个专家的副本数列表 `cnt`；
2. `balanced_packing(weights, n_packs)`：把 `len(weights)` 个物品分到 `n_packs` 个包里，**每包物品数相同**。按重量从大到小（重量相同时下标小的在前）依次处理，每个物品放进"还有空位的包里当前总重最小的那个"（并列时取编号小的包）。返回每个包的物品下标列表；
3. `imbalance(load, cnt, packs, phys)`：`phys` 是物理专家列表（第 `i` 个物理专家对应的逻辑专家编号），`packs` 是 `balanced_packing` 对物理专家的分组结果。每个物理专家的负载是 `load[e] / cnt[e]`，返回"最重的包 ÷ 平均包重"。

```python
cnt = replicate([8, 1, 1, 2], 6)          # [3, 1, 1, 1]
balanced_packing([5, 1, 4, 2], 2)         # [[0, 1], [2, 3]]
```

<!-- 题解 -->
复制解决"单个专家太热、放在哪都会成为瓶颈"的问题，打包解决"把热的和冷的搭配到一起"。两步都是贪心，但对 EPLB 足够好：它每隔几分钟根据统计重新计算一次，追求的是快而稳，而不是最优解。
分层版本就是把这两步套在"组 → 节点""节点内专家 → 卡"两级上：先用 `balanced_packing` 把专家组打包到节点，再在每个节点内 `replicate` 和打包。
