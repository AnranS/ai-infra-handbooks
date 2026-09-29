---
title: RL 损失的几个部件：平均方式、GSPO 比率、CISPO 权重与 k3
chapter: algo/rl-algorithms.md
difficulty: 中等
tags: [RL, GRPO, DAPO, GSPO, CISPO, KL]
---
GRPO 之后的几种 RL 算法，改的都是损失函数里的某个部件。每个回答是一个 token 列表，下面的函数都用纯 Python 实现：

1. `token_weights(advantages, lengths, mode)`：返回每个回答里**每个 token** 在损失中的权重（优势 × 系数）。`mode="seq"`（原始 GRPO）：先在每个回答内部平均、再对回答平均，系数是 $1 / (|o_i| \cdot N)$；`mode="token"`（DAPO）：所有 token 一起平均，系数是 $1 / \sum_i |o_i|$；
2. `gspo_ratio(logp_new, logp_old)`：两个等长的 log 概率列表（一个回答的所有 token），返回序列级的重要性比，即逐 token 比率的**几何平均** $\exp\big(\frac{1}{L}\sum_t (\log\pi_\text{new} - \log\pi_\text{old})\big)$；
3. `cispo_weights(ratios, eps_low=0.2, eps_high=0.28)`：把每个 token 的重要性比截断到 $[1 - \varepsilon_\text{low}, 1 + \varepsilon_\text{high}]$，返回截断后的权重（它在 CISPO 里不回传梯度，这里只算数值）；
4. `k3(logp, logp_ref)`：逐 token 的 KL 估计量 $k_3 = (r - 1) - \log r$，$r = \exp(\log p_\text{ref} - \log p)$，返回列表。

```python
token_weights([1.0, -1.0], [2, 4], "seq")     # [0.25, -0.125]：每个回答总权重相同
token_weights([1.0, -1.0], [2, 4], "token")   # [1/6, -1/6]：每个 token 权重相同
gspo_ratio([-1.0, -2.0], [-1.2, -1.8])        # exp((0.2 - 0.2) / 2) = 1.0
```

<!-- 题解 -->
按回答平均时，长回答的每个 token 权重更小：又长又错的回答受到的惩罚被摊薄，模型倾向于把错误的回答写长；token 平均让每个 token 同等对待。
几何平均是逐 token 比率乘积的 $1/L$ 次方：乘积随长度指数发散，几何平均始终在 1 附近，GSPO 才能在序列级上用很窄的裁剪范围。
CISPO 的权重被截断但不回传梯度，每个 token 的梯度 $\nabla\log\pi$ 都保留；PPO 式的裁剪会让超出范围的 token 完全没有梯度。
k3 在 $r = 1$（两个分布在这个 token 上一致）时为 0，其余时候为正，期望等于 KL，方差比 $-\log r$ 小得多。
