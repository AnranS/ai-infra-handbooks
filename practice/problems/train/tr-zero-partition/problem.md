---
title: ZeRO 的参数切分与显存、通信账本
chapter: data/zero-fsdp.md
difficulty: 中等
tags: [ZeRO, FSDP, 切分, 显存账本, 估算]
---
ZeRO 和 FSDP 把所有参数按顺序拼成一个一维缓冲区，补零到卡数 `N` 的整数倍，再均分成 `N` 段，rank `r` 负责第 `r` 段（它的优化器状态、梯度，ZeRO-3 时还有参数本身）。

1. `partition(sizes, world)`：`sizes[i]` 是第 `i` 个参数的元素个数。返回 `(per, segments)`：`per` 是每段的长度；`segments[r]` 是 rank `r` 负责的片段列表，每项 `(参数下标, 起点, 终点)`，起点和终点是**参数内部**的偏移（左闭右开），按顺序排列，补零的部分不列出；
2. `model_state_bytes(psi, world, stage)`：bf16 混合精度 + Adam（bf16 参数 2 字节、bf16 梯度 2 字节、fp32 主参数与两个矩共 12 字节），`psi` 个参数、`world` 张卡时，ZeRO 第 `stage` 级（0～3）每张卡的模型状态字节数；其他 `stage` 抛出 `ValueError`；
3. `comm_bytes(psi, world, stage)`：每一步每张卡发送的字节数（bf16，ring 算法）。不切分和 ZeRO-1/2：梯度的一次 all-reduce（或等价的 reduce-scatter + 参数 all-gather）；ZeRO-3：前向 all-gather 参数、反向再 all-gather 一次、梯度 reduce-scatter。

```python
partition([3, 5, 2], 4)
# (3, [[(0, 0, 3)], [(1, 0, 3)], [(1, 3, 5), (2, 0, 1)], [(2, 1, 2)]])
model_state_bytes(70e9, 64, 1) / 2**30     # 约 273 GiB：bf16 参数和梯度不切，只有优化器状态被切成 64 份
```

<!-- 题解 -->
一个参数可能跨两个 rank（上例的第 1、2 个参数），所以 ZeRO 的优化器要按"片段"而不是按"参数"来更新：rank 2 只更新第 1 个参数的后两个元素和第 2 个参数的第一个元素。FSDP1 就是这样平铺切分的；FSDP2 改成每个参数各自按第 0 维切（DTensor），代价是每个参数都要补零，好处是切分后的参数仍然是一个完整的张量，便于和 TP 等其他并行组合。

ZeRO-1、2 的通信量和普通数据并行相同（reduce-scatter + all-gather 恰好等于一次 all-reduce），ZeRO-3 多出 50%。
