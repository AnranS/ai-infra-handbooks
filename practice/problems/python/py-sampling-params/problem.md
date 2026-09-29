---
title: 带校验的 SamplingParams
chapter: core/data-classes.md
difficulty: 简单
tags: [dataclass, __post_init__, 不可变]
---
推理服务的每个请求都带一组采样参数。用 `dataclasses` 实现一个**不可变**的 `SamplingParams`：

| 字段 | 类型 | 默认值 | 合法范围 |
| --- | --- | --- | --- |
| `temperature` | `float` | `1.0` | `>= 0` |
| `top_p` | `float` | `1.0` | `(0, 1]` |
| `top_k` | `int` | `-1` | `-1`（不限制）或 `>= 1` |
| `max_tokens` | `int` | `16` | `>= 1` |
| `stop` | `tuple[str, ...]` | `()` | 构造时也接受 `list` 和单个 `str`，统一转成 `tuple` |

要求：

- 不合法的值在构造时抛出 `ValueError`，信息里要包含字段名；
- 实例不可变（给字段赋值抛出 `dataclasses.FrozenInstanceError`），并且**可哈希**；
- `greedy` 属性：`temperature == 0` 时为 `True`；
- `with_(**changes)` 返回修改了部分字段的新实例，新实例同样要经过校验。

```python
p = SamplingParams(temperature=0.7, stop=["</s>", "\n\n"])
p.stop                      # ('</s>', '\n\n')
p.with_(max_tokens=128)     # 新对象，其余字段不变
SamplingParams(top_p=0)     # ValueError: top_p ...
```

<!-- 题解 -->
`@dataclass(frozen=True)` 让实例不可变，同时自动生成基于字段的 `__hash__`。
`frozen` 的实例在 `__post_init__` 里不能直接 `self.stop = ...`，要用 `object.__setattr__(self, "stop", tuple(...))`。
`with_` 直接用 `dataclasses.replace(self, **changes)`：它调用 `__init__`，所以 `__post_init__` 的校验也会执行。
