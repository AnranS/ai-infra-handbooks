---
title: 用描述符做字段校验
chapter: types/metaprogramming.md
difficulty: 中等
tags: [描述符, __set_name__, 元编程]
---
写一组**描述符**，让配置类的字段在赋值时自动校验：

```python
class ModelConfig:
    hidden_size = Positive(int)
    num_heads = Positive(int)
    dropout = Bounded(float, 0.0, 1.0)
    dtype = OneOf("float16", "bfloat16", "float32")

    def __init__(self, hidden_size, num_heads, dropout=0.0, dtype="bfloat16"):
        self.hidden_size = hidden_size        # 赋值时自动校验
        ...

cfg = ModelConfig(1024, 16)
cfg.num_heads = 0          # ValueError: num_heads 必须是正数，收到 0
cfg.dtype = "int4"         # ValueError: dtype 必须是 ['float16', 'bfloat16', 'float32'] 之一，收到 'int4'
ModelConfig.hidden_size    # 通过类访问时返回描述符对象本身
```

要实现：

- 基类 `Field`：用 `__set_name__` 记住字段名（错误信息里要用），值存在实例的 `__dict__` 里（每个实例互不影响）；读取还没赋值的字段抛出 `AttributeError`；
- `Typed(tp)`：类型不对抛出 `TypeError`；`float` 字段也接受 `int`（存成 `float`），但任何字段都不接受 `bool`；
- `Positive(tp)`：在 `Typed` 基础上要求 `> 0`，否则 `ValueError`；
- `Bounded(tp, lo, hi)`：在 `Typed` 基础上要求 `lo <= v <= hi`；
- `OneOf(*choices)`：必须是其中之一。

所有错误信息都要包含字段名。

<!-- 题解 -->
数据描述符（定义了 `__set__`）的优先级高于实例字典，所以 `cfg.x = 1` 一定会走 `Field.__set__`；
值存在 `instance.__dict__[self.name]`，读取时 `__get__` 从这里取。`instance is None` 表示通过类访问，返回 `self`。

校验逻辑放在一个 `validate(value)` 方法里，子类重写并调用 `super().validate(value)`，这样 `Positive`、`Bounded` 复用了 `Typed` 的类型检查。
