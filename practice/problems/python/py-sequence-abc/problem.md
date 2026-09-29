---
title: 继承 Sequence，白拿一堆方法
chapter: types/protocols.md
difficulty: 中等
tags: [collections.abc, 协议, 切片]
---
实现一个只读的等差数列 `ArithSeq(start, stop, step=1)`，行为和 `range` 一致（`step` 可以为负，不能为 0），但元素可以是 `float`：

```python
s = ArithSeq(0, 1, 0.25)
list(s)          # [0, 0.25, 0.5, 0.75]
s[-1]            # 0.75
s[1:3]           # ArithSeq(0.25, 0.75, 0.25)，切片仍然是 ArithSeq
0.5 in s, s.index(0.5), s.count(0.5)    # True, 2, 1
list(reversed(ArithSeq(5, 0, -2)))      # [1, 3, 5]
```

要求：

- 继承 `collections.abc.Sequence`，**只实现** `__len__` 和 `__getitem__`（以及 `__init__`、`__repr__`、`__eq__`），`__contains__`、`__iter__`、`__reversed__`、`index`、`count` 都从 `Sequence` 继承；
- 第 `i` 个元素是 `start + i * step`（用乘法算，不要累加，避免误差累积）；
- 下标支持负数，越界抛出 `IndexError`；下标类型不对抛出 `TypeError`；
- 切片返回新的 `ArithSeq`（可以用 `slice.indices(len(self))`）；
- `repr` 形如 `ArithSeq(0.25, 0.75, 0.25)`；两个 `ArithSeq` 元素完全相同时相等；
- `isinstance(ArithSeq(0, 1), collections.abc.Sequence)` 为 `True`。

<!-- 题解 -->
长度：`max(0, ceil((stop - start) / step))`。

切片：`start_i, stop_i, step_i = key.indices(len(self))`，新序列的起点是 `self[start_i]` 对应的值 `start + start_i * step`，
步长 `step * step_i`，长度 `len(range(start_i, stop_i, step_i))`；终点取 `新起点 + 长度 * 新步长`，保证长度正确。

继承 `Sequence` 之后，`in`、迭代、`reversed`、`index`、`count` 都会调用 `__getitem__` 和 `__len__` 自动实现，这就是抽象基类的"混入方法"。
