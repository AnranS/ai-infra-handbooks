---
title: BaseOP：不用 nn.Module 收集和加载权重
chapter: compute/layers.md
difficulty: 中等
tags: [递归, 权重命名, 反射]
---
mini-sglang 没有用 `torch.nn.Module`：推理不需要自动求导和 hook，只需要"按名字收集和加载权重"。它的 `BaseOP` 直接遍历对象的 `__dict__`：

- **不以下划线开头**的数组属性是权重（这里用 numpy 数组代替张量）；
- `BaseOP` 类型的属性是子模块，名字按属性路径用 `.` 拼接；
- `OPList` 是子模块列表，名字是 `0`、`1`、`2`……（对应 checkpoint 里的 `layers.0`）；
- 以下划线开头的属性（缓存、配置）一律忽略。

实现 `BaseOP.state_dict(prefix="")`（返回 `{名字: 数组}`，按属性定义的顺序）和 `BaseOP.load_state_dict(state_dict)`：

- 加载时按名字取出对应的数组，**替换**属性（不是拷贝进去）；形状或 dtype 不一致时抛出 `ValueError`（信息里包含名字）；缺少某个名字时抛出 `KeyError`；
- 加载完成后如果 `state_dict` 里还有没用到的键，抛出 `KeyError`，列出这些键（只在最外层检查）；
- 不要修改调用者传进来的字典（先复制一份再 pop）；
- `OPList` 也要支持这两个方法。

```python
class Linear(BaseOP):
    def __init__(self, i, o): self.weight = np.zeros((o, i), np.float32)

class Block(BaseOP):
    def __init__(self):
        self.qkv_proj = Linear(4, 12)
        self.norm_weight = np.ones(4, np.float32)
        self._cache = np.zeros(3)                 # 忽略

class Model(BaseOP):
    def __init__(self): self.layers = OPList([Block(), Block()])

Model().state_dict().keys()
# ['layers.0.qkv_proj.weight', 'layers.0.norm_weight', 'layers.1.qkv_proj.weight', 'layers.1.norm_weight']
```

<!-- 题解 -->
递归：`state_dict` 遍历 `self.__dict__.items()`，数组直接放进结果，`BaseOP` 递归调用并传入拼好的前缀；`OPList` 用下标作为名字。

`load_state_dict` 用一个内部参数区分"最外层"和"递归调用"，只有最外层在结束时检查多余的键。
书中的模型先在 meta 设备上构造（没有真实内存），加载时直接替换引用，所以加载一个 7B 模型不需要先分配一份空权重。
