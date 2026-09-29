---
title: QK-Clip：按头压住注意力 logits
chapter: algo/stability.md
difficulty: 简单
tags: [训练稳定性, QK-Clip, 注意力, Muon]
---
Kimi K2 在每次优化器更新之后执行 QK-Clip：统计每个头在这批数据上的最大注意力 logit $S_{\max}$，超过阈值 $\tau$ 的头，把它的 $W_q$、$W_k$ 分别乘 $\gamma^{\alpha}$、$\gamma^{1-\alpha}$，$\gamma = \tau / S_{\max}$。矩阵用嵌套列表表示，不用 numpy。实现：

1. `max_logit(x, wq, wk)`：`x` 是 `T × D` 的输入，`wq`、`wk` 是一个头的 `D × d` 权重。`q = x·wq`、`k = x·wk`，返回所有 `(i, j)` 上 `q_i · k_j / √d` 的最大值（不加因果掩码）；
2. `qk_clip(x, Wq, Wk, tau, alpha=0.5)`：`Wq`、`Wk` 是每个头的权重列表。返回 `(新的 Wq, 新的 Wk, gammas)`，`gammas[h] = min(1, tau / S_max[h])`，没有超过阈值的头保持原样（`gamma = 1`）。不要修改传入的列表。

```python
x = [[1.0, 0.0], [0.0, 1.0]]
Wq, Wk = [[[20.0], [0.0]]], [[[10.0], [0.0]]]      # 一个头，d = 1：最大 logit = 20·10 = 200
new_q, new_k, g = qk_clip(x, Wq, Wk, tau=100.0)   # g = [0.5]，new_q = [[[20·√0.5], [0]]]
```

<!-- 题解 -->
logit 对 $W_q$、$W_k$ 都是线性的，所以两边各乘 $\gamma^{\alpha}$、$\gamma^{1-\alpha}$ 之后，这个头所有的 logits 恰好缩小 $\gamma$ 倍，最大值正好等于 $\tau$。按头处理，是因为各个头的 logits 规模差别很大，整层一起缩放会误伤正常的头。
Kimi K2 用 MLA，推理时 key 不按头显式算出来，没法在中间插 QK-Norm，这是它选择在训练中直接改权重的原因；对 MLA 只缩放每个头自己的那部分 q、k，所有头共享的 RoPE key 不动。
