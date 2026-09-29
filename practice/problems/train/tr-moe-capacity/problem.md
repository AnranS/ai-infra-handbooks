---
title: MoE 的容量、丢 token 与负载均衡损失
chapter: model/moe-ep.md
difficulty: 中等
tags: [MoE, 专家并行, 负载均衡, 容量因子]
requires: [numpy]
---
训练时每个专家的缓冲区大小是固定的（这样 all-to-all 和分组 GEMM 的形状才固定）：`T` 个 token、每个选 `k` 个专家、`E` 个专家时，每个专家最多接收 `capacity(T, k, E, factor) = ceil(factor · T · k / E)` 份，超出的**丢弃**（这个 token 在该专家上的贡献为 0，只剩残差）。

1. `capacity(T, k, E, factor)`；
2. `dispatch(topk_idx, topk_w, E, cap, policy="order")`：`topk_idx`、`topk_w` 都是 `(T, k)`。对每个专家，把选中它的 (token, 第几个选择) 按优先级排队，前 `cap` 份留下。`policy="order"` 按 token 编号（同一个 token 按第几个选择）；`policy="score"` 按门控权重从大到小（一样大时按 token 编号），即"按分数优先"。返回 `(T, k)` 的整数数组 `slot`：留下的是它在该专家缓冲区里的位置（排队的名次，从 0 开始），丢弃的是 `-1`；
3. `combine(expert_out, topk_idx, topk_w, slot)`：`expert_out` 是 `(E, cap, d)`，每个专家对缓冲区里每一份的输出。返回 `(T, d)`：每个 token 把留下的那几份按门控权重加权求和；
4. `aux_loss(logits, topk_idx)`：Switch Transformer 的负载均衡损失 `E · Σ_e f_e · P_e` 及其对 `logits` 的梯度。`P_e` 是 `softmax(logits)` 在所有 token 上的平均，`f_e` 是选中专家 `e` 的份数占全部 `T · k` 份的比例（`f` 不可导，当作常数）。返回 `(loss, dlogits)`。

```python
topk_idx = np.array([[0, 1], [0, 2], [0, 1]])
topk_w = np.array([[0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
dispatch(topk_idx, topk_w, E=3, cap=2)                   # [[0, 0], [1, 0], [-1, 1]]：专家 0 满了，第 3 个 token 被丢
dispatch(topk_idx, topk_w, E=3, cap=2, policy="score")   # [[-1, 0], [1, 0], [0, 1]]：按分数，丢的是权重最小的 0.6
```

<!-- 题解 -->
`dispatch` 的核心是"按专家分组、组内按优先级排名"：把所有份按 `(专家, 优先级)` 排序，名次就是组内的下标。`combine` 按 `slot` 把输出取回来，丢掉的份不参与加权——这个 token 的 MoE 输出变小了，只能靠残差传下去。容量因子越大丢得越少，但缓冲区（显存、all-to-all 的数据量、GEMM 的补零）越大；训练常用 1.0～1.25，推理为了不丢 token 通常不设上限（DeepSeek-V3 训练时也不丢 token，靠负载均衡保证不爆）。

辅助损失的梯度：$\partial L / \partial P_e = E f_e$，再乘 softmax 的雅可比：对每个 token，$\frac{\partial L}{\partial z_t} = \frac{1}{T}\, p_t \odot (a - p_t \cdot a)$，其中 $a_e = E f_e$。它让被选得多（$f_e$ 大）的专家的分数下降——这就是"辅助损失和主目标抢梯度"的来源，也是 DeepSeek-V3 改用偏置调整的原因（见本章）。
