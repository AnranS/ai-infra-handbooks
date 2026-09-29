---
title: 反向引擎：依赖计数、梯度累加与广播的还原
chapter: framework/autograd.md
difficulty: 困难
tags: [autograd, 计算图, 广播, 拓扑排序]
requires: [numpy]
---
模板里是一个张量级的迷你 autograd：`Tensor` 上的 `+`、`*`、`@`、`sum()`、`relu()` 在前向时创建反向节点（`Node` 的子类），节点记着前向的输入 `inputs`，`backward(grad)` 返回对每个输入的梯度。模板的 `Tensor.backward` 是递归写法：梯度每到一个节点就立刻往下传——结果对，但一个节点被多条路径用到时会被调用很多次，深一点的图会指数爆炸。实现：

1. `unbroadcast(grad, shape)`：前向时输入被广播过（比如 `(3, 1) * (4,)` 得到 `(3, 4)`），梯度要**求和**还原成输入的形状：多出来的前导维度求和去掉，输入里长度为 1、梯度里不为 1 的维度求和并保留（`keepdims`）。模板里的 `AddBackward`、`MulBackward` 已经调用它；
2. `Tensor.backward(grad=None)`：像 PyTorch 的引擎那样执行反向（`grad` 为 `None` 时用全 1）：
   - 先从输出出发遍历一遍图，数出每个节点有几条"入边"（它的输出被几个节点当作输入用了，同一个节点用两次算两条）；
   - 维护每个节点收到的梯度之和；一个节点的入边**全部**到齐之后才调用它的 `backward`，**每个节点只调用一次**；
   - 叶子张量（没有 `grad_fn`、`requires_grad=True`）把梯度**累加**到 `.grad` 上（多次调用 `backward` 会一直累加）；中间结果的 `.grad` 保持 `None`；`requires_grad=False` 的张量不接收梯度。

```python
x = Tensor([1.0, 2.0], requires_grad=True)
y = Tensor([3.0, 4.0], requires_grad=True)
(x * y + x).sum().backward()
x.grad, y.grad        # [4., 5.]、[1., 2.]
```

<!-- 题解 -->
依赖计数就是拓扑排序的 Kahn 算法：入边计数为 0 的节点进入就绪队列，执行后给它的输入节点减一。PyTorch 的 `torch/csrc/autograd/engine.cpp` 正是这样：先 `compute_dependencies` 数出每个节点的依赖，再用 `InputBuffer` 累加同一个节点收到的多份梯度，`ReadyQueue` 里是依赖已经满足的节点。每个节点只执行一次，总代价与图的大小成正比。

`unbroadcast` 是很多手写反向的 bug 来源：偏置 `b` 的形状是 `(d,)`，`x + b` 的梯度是 `(B, d)`，对 `b` 的梯度要沿 batch 维求和。叶子累加梯度而不是覆盖，是"每一步要 `zero_grad()`"的原因，也是梯度累积（多个 micro-batch 反向之后再更新）能直接实现的原因。
