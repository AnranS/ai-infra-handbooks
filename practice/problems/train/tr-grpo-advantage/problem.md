---
title: GRPO 的组内优势与裁剪损失
chapter: practice/frameworks-rl.md
difficulty: 简单
tags: [RL, GRPO, PPO, 优势, 动态采样]
---
GRPO 对每个 prompt 采样 `G` 条回答（一组），用组内奖励的均值做基线，不需要价值模型。实现：

1. `group_advantages(rewards, G, eps=1e-6)`：`rewards` 按组连续排列（前 `G` 个是第一组）。每条回答的优势是 $(r - \text{组均值}) / (\text{组标准差} + \varepsilon)$，标准差用**样本标准差**（除以 $G - 1$，和 `torch.std` 的默认一致）。`len(rewards)` 不是 `G` 的整数倍、或 `G < 2` 时抛出 `ValueError`；
2. `clip_loss(ratios, advantages, eps=0.2)`：PPO 式的裁剪目标，逐 token 计算 $\min(\rho A,\ \text{clip}(\rho, 1-\varepsilon, 1+\varepsilon) A)$，返回它们平均值的**相反数**（要最小化的损失）；
3. `informative_groups(rewards, G)`：返回组内奖励**不全相同**的组的下标（从 0 开始）。全对或全错的组优势全为 0，没有学习信号。

```python
group_advantages([1, 0, 1, 0], 2)          # 约 [0.707, -0.707, 0.707, -0.707]
informative_groups([1, 1, 0, 1, 0, 0], 2)  # [1]
```

<!-- 题解 -->
组内归一化让不同难度的题目有可比的信号：一道大家都做对的题，做对不再得到奖励；一道很难的题，偶尔做对一次就有很大的正优势。
全对、全错的组对梯度没有贡献，却占了 rollout 的算力。DAPO 的"动态采样"就是在采样后过滤掉这些组，并继续采样直到凑满一个 batch；这让 rollout 的工作量变得不确定，也是 RL 系统里长尾和异步问题的来源之一。
裁剪损失里，$\rho$ 的分母是训练端**重算**的旧策略概率：第一次更新前 $\rho = 1$，裁剪只约束"这一批数据上更新了几次之后策略变了多少"。
