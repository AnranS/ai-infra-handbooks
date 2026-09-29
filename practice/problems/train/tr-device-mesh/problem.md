---
title: device mesh：并行维度的排布与通信组
chapter: practice/strategy.md
difficulty: 中等
tags: [3D 并行, device mesh, Megatron, 通信组]
---
把所有卡看成一个多维数组，每一维对应一种并行（tp、cp、dp、pp……），rank 号按"外层维度在前、最内层维度变化最快"排列；沿某一维的每一条"线"上的卡组成这种并行的一个通信组。实现：

1. `mesh_groups(sizes, order)`：`sizes` 是 `{维度名: 度数}`，`order` 是维度名的列表，**从外到内**。返回 `{维度名: 通信组列表}`，每个组是按 rank 升序的列表，组与组之间按组里最小的 rank 升序；
2. `coords(rank, sizes, order)`：返回这个 rank 在各维上的坐标 `{维度名: 坐标}`；
3. `megatron_order(spec, sizes)`：Megatron 用 `"tp-cp-ep-dp-pp"` 这样的字符串**从内到外**描述顺序；返回从外到内的 `order`，并去掉 `sizes` 里没有的维度（比如 ep 借用 dp 的卡，不单独占一维）；
4. `intra_node(groups, gpus_per_node)`：是否每个组的卡都在同一个节点里（rank 整除 `gpus_per_node` 得到节点号）。

```python
sizes = {"tp": 2, "cp": 2, "dp": 2, "pp": 2}
g = mesh_groups(sizes, ["pp", "dp", "cp", "tp"])
g["cp"][:2]                   # [[0, 2], [1, 3]]
coords(13, sizes, ["pp", "dp", "cp", "tp"])    # {'pp': 1, 'dp': 1, 'cp': 0, 'tp': 1}
```

<!-- 题解 -->
rank 就是各维坐标的"混合进制"数：最内层的步长是 1，往外每一层的步长乘上内层的度数。`coords` 是反过来拆：从最内层开始取余、整除。一个维度的通信组，就是固定其他维的坐标、让这一维从 0 走到度数 − 1 得到的那些 rank。

最内层的维度 rank 号相邻，落在同一个节点里，所以 TP 放在最内层：它每层前向、反向都有 all-reduce，只有 NVLink 扛得住。16 张卡、每节点 8 张时，按 `pp-dp-cp-tp`（从外到内）排，TP、CP、DP 组都在节点内，只有 PP 跨节点。Llama 3 的顺序是（从内到外）`[TP, CP, PP, DP]`，把 DP（FSDP）放在最外层，因为 FSDP 的通信可以提前发起、最能容忍跨节点的延迟。
