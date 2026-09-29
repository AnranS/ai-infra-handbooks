---
title: 让 Vector 像内置类型一样好用
chapter: core/data-model.md
difficulty: 简单
tags: [特殊方法, 运算符重载, 可哈希]
---
实现一个二维向量类 `Vector(x, y)`，通过特殊方法让它支持这些操作：

| 表达式 | 结果 |
| --- | --- |
| `repr(Vector(1, 2))` | `'Vector(1, 2)'`（`x`、`y` 用各自的 `repr`） |
| `Vector(1, 2) == Vector(1, 2)` | `True`；和非 `Vector` 比较返回 `NotImplemented` |
| `hash(v)` | 相等的向量哈希值相同，可以放进 `set`、当 `dict` 的键 |
| `abs(Vector(3, 4))` | `5.0` |
| `bool(Vector(0, 0))` | `False`（零向量为假） |
| `v + w`、`v - w` | 逐分量相加减，得到新的 `Vector` |
| `v * 3`、`3 * v` | 数乘 |
| `x, y = v` | 支持解包（实现 `__iter__`） |
| `-v` | 取反 |

向量是**不可变**的：`x`、`y` 是只读属性，给 `v.x` 赋值要抛出 `AttributeError`。

<!-- 题解 -->
- 只读属性：把值存在 `_x`、`_y`，用 `@property` 暴露；没有 setter 时赋值会抛 `AttributeError`。也可以直接继承 `typing.NamedTuple`。
- `__eq__` 遇到不认识的类型返回 `NotImplemented`（不是 `False`），让 Python 有机会尝试另一边的 `__eq__`。
- 定义了 `__eq__` 的类默认 `__hash__ = None`（不可哈希），要自己实现 `__hash__`，并保证"相等则哈希相同"：`hash((self.x, self.y))`。
- `3 * v` 调用的是 `v.__rmul__(3)`。
