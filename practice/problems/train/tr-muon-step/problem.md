---
title: Muon 的一步：Newton-Schulz 正交化
chapter: algo/optimizer.md
difficulty: 中等
tags: [优化器, Muon, Newton-Schulz, 矩阵运算]
---
Muon 对隐藏层的每个权重矩阵维护动量，更新时把动量矩阵"正交化"（奇异值都推到 1 附近）再乘学习率。矩阵用嵌套列表表示（`G[i][j]`），不要用 numpy。实现：

1. `newton_schulz(G, steps=5)`：
   - 先除以 Frobenius 范数：`X = G / (‖G‖_F + 1e-7)`；
   - 行数大于列数时先转置，保证 `X Xᵀ` 是较小的方阵，最后再转置回来；
   - 迭代 `steps` 次：`A = X Xᵀ`，`X ← a·X + (b·A + c·A·A)·X`，系数 `a, b, c = 3.4445, -4.7750, 2.0315`；
2. `muon_step(W, grad, buf, lr, momentum=0.95, weight_decay=0.0)`，返回 `(新的 W, 新的 buf)`：
   - `buf ← momentum·buf + grad`；
   - `U = newton_schulz(grad + momentum·buf)`（Nesterov 式，用更新后的 `buf`）；
   - `W ← W·(1 - lr·weight_decay) - lr·0.2·√max(行数, 列数)·U`。

```python
X = newton_schulz([[3.0, 0.0], [0.0, 0.1]])   # 对角矩阵：奇异值 3 和 0.1 相差 30 倍
# X ≈ [[0.697, 0], [0, 1.129]]：两个都被推到 0.7～1.2 之间
```

<!-- 题解 -->
Newton-Schulz 只用矩阵乘，GPU 上又快又能用 bf16；五次多项式的系数是为"收敛快"挑的：小的奇异值每步放大约 3.4 倍，很快进入 1 附近，之后在约 0.7～1.2 之间来回跳（例子里多迭代几步就能看到），不会收敛到精确的 1。对优化来说这已经足够。
先除以 Frobenius 范数，是为了让所有奇异值都不超过 1，迭代才会收敛；转置是为了让 `X Xᵀ` 是较小的方阵，省计算。
`0.2·√max(行, 列)` 让更新的均方根约为 0.2，和 AdamW 相当，于是可以沿用 AdamW 的学习率和权重衰减（Moonshot 的做法）。
