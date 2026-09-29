---
title: 用 __init_subclass__ 做插件注册表
chapter: core/oop.md
difficulty: 中等
tags: [继承, __init_subclass__, 类属性]
---
推理框架里常见"按名字创建后端"：`create_backend("flashinfer")`。实现一个基类 `Backend`，让子类在**定义时**自动注册：

```python
class FlashInfer(Backend, name="flashinfer"):
    def forward(self, x): return x + 1

class Torch(Backend, name="torch", priority=10):
    def forward(self, x): return x * 2

Backend.create("torch").forward(3)     # 6
Backend.available()                    # ["torch", "flashinfer"]
```

要求：

- 子类用类参数 `name=...`（必填）和 `priority=...`（可选，默认 0）注册；`name` 重复时抛出 `ValueError`；
- 没有传 `name` 的子类是"抽象的中间类"，不注册，但它的子类仍然可以注册；
- `Backend.create(name, *args, **kwargs)`：创建实例，名字不存在时抛出 `KeyError`，错误信息里列出所有可用的名字；
- `Backend.available()`：按 `priority` 从高到低、同优先级按名字排序返回名字列表；
- 每个注册的子类都有类属性 `name`；
- 为了让测试互不影响，提供 `Backend.clear_registry()` 清空注册表。

<!-- 题解 -->
`__init_subclass__(cls, **kwargs)` 在每个子类创建时被调用（隐式的类方法），类定义里的关键字参数会传给它。
注册表放在基类 `Backend` 上（`Backend._registry`，不是 `cls._registry`，否则会在子类上建新的字典）。
记得调用 `super().__init_subclass__(**kwargs)`，让多重继承时其他基类也能收到参数。
