---
title: 手写自动微分（mini autograd）
chapter: math/calculus.md
difficulty: 困难
tags: [反向传播, 计算图, 链式法则]
---
实现一个标量自动微分引擎，理解 PyTorch 的 `backward()` 在做什么。

`Value(data)` 包装一个 Python 浮点数，支持：

- 运算：`+`、`-`、`*`、`/`、`**`（指数是 Python 数字）、取负，以及和普通数字混合运算（`2 * v`、`v + 1`、`1 / v`）；
- 函数：`v.exp()`、`v.log()`、`v.tanh()`、`v.relu()`；
- `v.backward()`：从 `v` 出发做反向传播，把 $\partial v / \partial x$ 累加到每个上游节点的 `x.grad`（`v.grad` 本身设为 1）。

要求：

1. 一个节点被多次使用时（例如 `y = x * x`），梯度要**累加**；
2. 反向传播要按**拓扑序**进行：每个节点的梯度收集完整之后才往上游传；
3. 计算图可能很深（几千层），不能用递归（会超过 Python 的递归深度限制）。

```python
a, b = Value(2.0), Value(-3.0)
c = a * b + a.exp()
c.backward()
a.grad        # b + exp(a) = -3 + 7.389... = 4.389...
b.grad        # a = 2.0
```

<!-- 题解 -->
每个 `Value` 记住它的父节点 `_prev` 和一个 `_backward` 闭包：闭包知道本节点的局部导数，把 `self.grad * 局部导数` 累加到父节点的 `grad` 上。
例如乘法 `out = a * b`：`a.grad += b.data * out.grad`、`b.grad += a.data * out.grad`（用 `+=` 保证多次使用时累加）。

`backward()`：先用**迭代式** DFS 求拓扑序（显式栈，记录"子节点都访问完了"再加入序列），然后倒序调用每个节点的 `_backward`。
这正是 PyTorch autograd 引擎的工作方式，只是那里的"值"是张量。
