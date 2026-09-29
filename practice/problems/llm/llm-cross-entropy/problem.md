---
title: 交叉熵、KL 散度与困惑度
chapter: math/information-theory.md
difficulty: 简单
tags: [交叉熵, KL 散度, ignore_index]
---
用 numpy 实现语言模型训练和评估里最常用的三个量：

1. `cross_entropy(logits, targets, ignore_index=-100)`：`logits` 形状 `(N, V)`，`targets` 形状 `(N,)`。
   返回所有**未被忽略**位置的平均负对数似然 $-\frac{1}{M}\sum \log p(y_i)$（$M$ 是未被忽略的位置数）；全部被忽略时返回 `0.0`。
   和 PyTorch 的 `F.cross_entropy` 一致，`targets == ignore_index` 的位置不参与计算（这些位置的 target 值可能越界，不能拿去索引）。
2. `kl_divergence(p_logits, q_logits)`：两组 logits 各自 softmax 得到 $P$、$Q$（形状 `(N, V)`），返回每一行的 $\mathrm{KL}(P \,\|\, Q) = \sum_v P_v (\log P_v - \log Q_v)$，形状 `(N,)`。
3. `perplexity(logits, targets, ignore_index=-100)`：交叉熵的指数。

都要数值稳定（`logits` 里可能有很大的数）。

<!-- 题解 -->
先算 `log_softmax`（减最大值），交叉熵就是 `-log_probs[mask][arange, targets[mask]]` 的平均值。
KL 用 log 概率相减：`(p * (log_p - log_q)).sum(-1)`，这样 $P_v$ 很小时也不会出现 `0 * log(0)` 的 nan。
知识蒸馏的损失、RLHF 里约束策略不要偏离参考模型的惩罚项，都是 KL 散度。
