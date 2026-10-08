---
title: autograd：中间结果的梯度、梯度累积与冻结
chapter: autograd.md
difficulty: 中等
tags: [autograd, retain_grad, 梯度累积, requires_grad]
requires: [torch]
---
三件事，分别对应 autograd 最常被问到的三个点：

1. `mid_grad(x)`：令 $h = \sigma(x)$、$y = \sum h^2$，返回 `(dy/dh, dy/dx)` 两个张量。注意 `h` 不是叶子节点，默认拿不到它的梯度；
2. `accumulated_grad(w, xs, ys)`：`w` 是形状 `(D,)` 的参数（`requires_grad=True`），`xs` 是 `n` 个形状 `(m, D)` 的小批、`ys` 是 `n` 个形状 `(m,)` 的标签。损失是 `mse = mean((x @ w - y) ** 2)`。要求**分 n 次 backward**（每次只能看到一个小批），最后 `w.grad` 必须等于把所有小批拼成一个大批算出来的梯度。返回 `w.grad`；
3. `freeze_count(linear_layers, trainable_prefix)`：输入一组 `nn.Linear`（列表）和一个字符串前缀，把名字不以该前缀开头的层的参数 `requires_grad` 置为 `False`，返回仍然可训练的**参数张量个数**。层的名字就是 `f"layer{i}"`，`i` 从 0 开始。

```python
x = torch.tensor([0.5, -1.0])
dh, dx = mid_grad(x)      # dh = 2 * sigmoid(x)
```

<!-- 题解 -->
**中间结果的梯度**。`h` 是计算出来的，不是叶子，`h.grad` 默认是 `None`（而且不报错，只给一个 UserWarning）。要拿到它，在 backward 之前调用 `h.retain_grad()`。这是「`.grad` 是 None」三种情况里最常遇到的一种 —— 另外两种是根本没开 `requires_grad`，和在 `no_grad` 里算的、压根没建图。手算对一遍：$\partial y/\partial h = 2h$，$\partial y/\partial x = 2h\,\sigma'(x) = 2h\,h(1-h)$。

**梯度累积**。梯度是**累加**进 `.grad` 的，所以分几次 backward 不用做任何特殊处理 —— 真正的坑是**缩放**。损失用的是 `mean`，每个小批的 mean 是在 `m` 个样本上取的，而大批的 mean 是在 `n*m` 个上取的，所以每个小批的损失要再除以 `n`，加起来才等于大批。小批大小不一样时，正确的做法是按样本数加权（`loss * m_i / 总样本数`）。这就是训练框架里 `loss / accum_steps` 那一行的由来，写错了相当于把学习率放大 n 倍。

**冻结**。`requires_grad = False` 只是不给这些参数算梯度，它们**还在优化器的参数组里**：带 weight decay 时仍然会被衰减，所以建优化器时应该只传可训练的那部分（`filter(lambda p: p.requires_grad, model.parameters())`）。另一种「冻结」是在前向里 `detach()`，那切的是**数据流**，上游所有层都收不到梯度了 —— 两者完全不同，微调 backbone 用前者。
