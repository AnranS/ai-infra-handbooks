---
title: 闭包与函数组合
chapter: core/functions.md
difficulty: 简单
tags: [闭包, nonlocal, 高阶函数]
---
实现三个小工具：

1. `make_counter(start=0, step=1)`：返回一个函数，每调用一次返回下一个计数值（第一次返回 `start`）。不同的计数器互不影响。
2. `compose(*fns)`：返回函数的组合，`compose(f, g, h)(x) == f(g(h(x)))`。不传参数时返回恒等函数。
3. `make_multipliers(n)`：返回长度为 `n` 的函数列表，第 `i` 个函数把参数乘以 `i`。

```python
c = make_counter(10, 5)
c(), c(), c()                       # 10, 15, 20
compose(str, abs, int)("-42")       # "42"
[f(3) for f in make_multipliers(4)] # [0, 3, 6, 9]
```

注意第 3 个：在循环里写 `lambda x: x * i`，所有函数都会乘以**最后一个** `i`（闭包捕获的是变量，不是值）。

<!-- 题解 -->
- 计数器：闭包里用 `nonlocal` 修改外层变量；
- 组合：`functools.reduce` 从右往左套，或者先反转列表再循环调用；
- 乘法器：用默认参数在定义时"固定"当前值：`lambda x, i=i: x * i`，或者 `functools.partial(operator.mul, i)`。
