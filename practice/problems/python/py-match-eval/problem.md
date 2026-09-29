---
title: 用 match 写一个表达式求值器
chapter: core/pattern-matching.md
difficulty: 中等
tags: [match, 结构化模式匹配, 递归]
---
表达式用嵌套的元组表示：

| 形式 | 含义 |
| --- | --- |
| 整数或浮点数 | 常量 |
| 字符串 `"x"` | 变量，从 `env` 里取值；不存在时抛出 `NameError` |
| `("neg", e)` | `-e` |
| `("add", a, b)`、`("sub", a, b)`、`("mul", a, b)`、`("div", a, b)` | 四则运算；除数为 0 时抛出 `ZeroDivisionError` |
| `("max", e1, e2, ...)` | 至少一个参数 |
| `("let", name, value, body)` | 先算 `value`，在 `body` 里把 `name` 绑定为它（不影响外层 `env`） |
| `("if", cond, then, else_)` | `cond` 非零时求 `then`，否则求 `else_`（只求其中一个分支） |

实现 `evaluate(expr, env=None)`，其他任何形式都抛出 `ValueError`（信息里带上出错的表达式）。

```python
evaluate(("add", 1, ("mul", "x", 3)), {"x": 2})         # 7
evaluate(("let", "y", 10, ("max", "y", 3, ("neg", 20))))  # 10
evaluate(("if", 0, ("div", 1, 0), 5))                     # 5（没有求 then 分支）
```

注意 `True`、`False` 也是 `int`，不能当作常量接受。

<!-- 题解 -->
`match` 的几个关键写法：

- 类模式 `case int() | float() if not isinstance(expr, bool)` 匹配数字，并用守卫排除布尔值；
- 序列模式 `case ("add" | "sub" | "mul" | "div" as op, a, b)` 匹配固定长度并绑定变量；
- `case ("max", first, *rest)` 匹配"至少一个"的变长序列；
- `case str(name)` 匹配字符串并绑定。

`let` 用 `{**env, name: value}` 生成新环境，不修改外层字典。
