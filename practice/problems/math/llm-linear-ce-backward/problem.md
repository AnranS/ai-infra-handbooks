---
title: 手推 LM head + 交叉熵的反向传播
chapter: calculus.md
difficulty: 中等
tags: [反向传播, 矩阵求导, 梯度检查]
---
语言模型最后一步：隐藏状态 $H \in \mathbb{R}^{N \times d}$ 乘以输出矩阵 $W \in \mathbb{R}^{d \times V}$ 加偏置 $b$ 得到 logits，再和目标 token 算平均交叉熵：

$$Z = HW + b,\qquad L = \frac{1}{N}\sum_{i} -\log \mathrm{softmax}(Z_i)_{y_i}$$

实现 `forward_backward(H, W, b, y)`，返回 `(loss, dH, dW, db)`：损失值和三个参数的梯度，形状分别与 `H`、`W`、`b` 相同。
不能用自动微分，要手推公式；不能用 Python 循环遍历样本（用矩阵运算）。

提示：$\dfrac{\partial L}{\partial Z} = \dfrac{1}{N}\big(\mathrm{softmax}(Z) - \mathrm{onehot}(y)\big)$。

<!-- 题解 -->
```python
Z = H @ W + b
Z = Z - Z.max(axis=1, keepdims=True)         # 减最大值保证稳定
lse = log(exp(Z).sum(axis=1, keepdims=True))
loss = -(Z[arange(N), y] - lse[:, 0]).mean() # 用 log-softmax，概率极小时也不会 log(0)
dZ = exp(Z - lse); dZ[arange(N), y] -= 1; dZ /= N
dW = H.T @ dZ                        # (d, N) @ (N, V)
db = dZ.sum(axis=0)
dH = dZ @ W.T                        # (N, V) @ (V, d)
```

矩阵求导的形状检查法：`dW` 必须和 `W` 同形状 `(d, V)`，能由 `H`（`(N, d)`）和 `dZ`（`(N, V)`）拼出来的只有 `H.T @ dZ`。
实际训练框架会把这一步和 softmax 融合，避免存下 `(N, V)` 的概率矩阵（V 可能有 15 万）。
