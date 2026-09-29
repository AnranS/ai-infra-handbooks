---
title: DPO 损失与 GRPO 组内优势
chapter: training/post-training.md
difficulty: 中等
tags: [DPO, GRPO, 对齐]
---
后训练里两个核心公式，用 numpy 实现。

**1. 序列对数概率** `sequence_logprob(logits, tokens, mask)`：`logits` 形状 `(B, T, V)`，是模型在每个位置对**下一个** token 的预测；
`tokens` 形状 `(B, T)`；`mask` 形状 `(B, T)`，为 `True` 的位置才计入（通常只算回答部分，不算提示词）。
位置 $t$ 的 logits 预测的是 `tokens[:, t+1]`，所以序列对数概率是 $\sum_{t=0}^{T-2} \text{mask}[t+1] \cdot \log \mathrm{softmax}(\text{logits}[t])[\text{tokens}[t+1]]$。返回形状 `(B,)`。

**2. DPO 损失** `dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1)`：四个参数都是形状 `(B,)` 的序列对数概率（策略模型、参考模型分别对"好回答""坏回答"），

$$\mathcal{L} = -\frac{1}{B}\sum \log \sigma\Big(\beta\big[(\pi_c - r_c) - (\pi_r - r_r)\big]\Big)$$

要数值稳定（括号里的值很大或很小都不能溢出）。返回 `(loss, reward_acc)`，`reward_acc` 是隐式奖励差 $(\pi_c - r_c) - (\pi_r - r_r) > 0$ 的样本比例。

**3. GRPO 优势** `grpo_advantages(rewards, group_size, eps=1e-6)`：`rewards` 是一维数组，每连续 `group_size` 个是同一个提示词的多个采样。
每组内做标准化：$A_i = (r_i - \mathrm{mean}) / (\mathrm{std} + \epsilon)$（`std` 用**总体**标准差，即 `np.std` 的默认值）。

<!-- 题解 -->
- 序列对数概率：`log_softmax(logits[:, :-1])`，用 `np.take_along_axis` 取出 `tokens[:, 1:]` 对应的值，乘 `mask[:, 1:]` 再求和。错位一格是最常见的 bug；
- $\log\sigma(z) = -\log(1 + e^{-z}) = -\mathrm{logaddexp}(0, -z)$，用 `np.logaddexp` 就不会溢出；
- GRPO：`r.reshape(-1, group_size)`，按行减均值、除以标准差。组内全部答对（或全错）时标准差为 0，优势全为 0，这一组对梯度没有贡献。
