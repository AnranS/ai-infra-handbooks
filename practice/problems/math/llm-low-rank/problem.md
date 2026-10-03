---
title: 低秩近似与 LoRA 的参数账
chapter: linear-algebra.md
difficulty: 中等
tags: [SVD, 低秩, LoRA]
---
LoRA 假设微调带来的权重更新 $\Delta W$ 是低秩的：$\Delta W \approx B A$，$B \in \mathbb{R}^{m \times r}$，$A \in \mathbb{R}^{r \times n}$。实现三个函数：

1. `low_rank(W, r)`：返回 $(B, A)$，使 $BA$ 是 $W$ 在 Frobenius 范数下**最好的**秩 $r$ 近似（Eckart–Young 定理：取 SVD 的前 $r$ 个奇异值）。要求 $B = U_r \sqrt{\Sigma_r}$、$A = \sqrt{\Sigma_r} V_r^\top$（奇异值平均分到两边）。
2. `rank_for_energy(W, ratio)`：最小的 $r$，使前 $r$ 个奇异值的平方和占全部的比例 $\ge$ `ratio`（`0 < ratio <= 1`）。
3. `lora_params(shapes, r)`：给定若干个权重矩阵的形状 `[(m, n), ...]`，返回 `(原参数量, LoRA 参数量, 比例)`，LoRA 对每个矩阵新增 $r(m + n)$ 个参数，比例 = LoRA 参数量 / 原参数量。

```python
W = np.random.randn(64, 32) @ np.random.randn(32, 48)    # 秩 32
B, A = low_rank(W, 8)          # B: (64, 8)，A: (8, 48)
lora_params([(4096, 4096)] * 4, r=16)   # (67108864, 524288, 0.0078125)
```

<!-- 题解 -->
`U, S, Vt = np.linalg.svd(W, full_matrices=False)`，`s = np.sqrt(S[:r])`，`B = U[:, :r] * s`，`A = s[:, None] * Vt[:r]`。

近似误差 $\|W - BA\|_F^2 = \sum_{i > r} \sigma_i^2$，所以"保留多少能量"只看奇异值：`np.cumsum(S**2) / np.sum(S**2)` 第一次 $\ge$ ratio 的位置 + 1。
注意浮点误差：比较时给一点容差（例如 `>= ratio - 1e-12`），否则 ratio=1 时可能因为舍入找不到。
