---
title: nn.Module：ModuleList、buffer 与 state_dict
chapter: module.md
difficulty: 中等
tags: [nn.Module, ModuleList, register_buffer, state_dict, load_state_dict]
requires: [torch]
---
`nn.Module` 真正的职责是**管参数**。把下面三件事写出来，每一件都对应一个「代码能跑但结果不对」的坑：

1. `class Stack(nn.Module)`：`Stack(dim, n_layers)` 里有 `n_layers` 个 `nn.Linear(dim, dim, bias=True)`，必须能被 `parameters()` 找到；另外注册一个名叫 `scale` 的 **buffer**，值是标量张量 `dim ** -0.5`（不训练，但要随 `state_dict` 存取、随 `.to()` 搬家）。`forward(x)` 逐层过 `linear` 再 `relu`，最后整体乘 `scale`；
2. `state_keys(model)`：返回 `model.state_dict()` 的 key 列表（保持原顺序）；
3. `load_partial(model, state)`：用 `strict=False` 加载，返回 `(missing, unexpected)` 两个**排好序的列表**。

```python
m = Stack(4, 3)
len(list(m.parameters()))     # 6 = 3 层 x (weight + bias)
state_keys(m)[:3]             # ['scale', 'layers.0.weight', 'layers.0.bias']
```

<!-- 题解 -->
**子模块必须放进 `nn.ModuleList`**。写成 `self.layers = [nn.Linear(...) for _ in range(n)]` 时代码照样跑、loss 照样降（别的层在学），但这些层的参数**一个都没注册**：`parameters()` 里没有、优化器找不到、`.to(device)` 搬不走、`state_dict` 里也没有。`len(list(m.parameters())) == 0` 是最快的自检方式，跑一行就知道。顺带一提，`super().__init__()` 必须是 `__init__` 的第一句，不然后面给属性赋 `nn.Parameter` 会直接报错。

**buffer 不是参数**。`self.register_buffer("scale", torch.tensor(dim ** -0.5))` 之后，它进 `state_dict`、跟着 `.to()` 走，但不在 `parameters()` 里、优化器不会更新它。BatchNorm 的滑动统计量、RoPE 的 cos/sin 表都是这么存的。要是直接写成普通属性 `self.scale = torch.tensor(...)`，在 CPU 上测没问题，一上 GPU 就是 `Expected all tensors to be on the same device`。

**`load_state_dict` 的返回值是用来读的**。`strict=False` 时它返回一个有 `missing_keys` 和 `unexpected_keys` 的具名元组：前者是模型里有、文件里没有（新加的层），后者相反（删掉的层）。微调时常见的做法是 backbone 用 `strict=False` 加载、新 head 单独初始化，并且**把返回值打进日志** —— 否则改错一个名字就会悄悄跳过一个权重，你只会看到结果莫名其妙地差。
