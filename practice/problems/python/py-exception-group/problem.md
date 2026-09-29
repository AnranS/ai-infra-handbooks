---
title: ExceptionGroup 与 except*
chapter: practice/whats-new.md
difficulty: 中等
tags: [ExceptionGroup, except*, Python 3.11]
---
批量校验请求时，希望一次报告**所有**错误，而不是遇到第一个就停下。Python 3.11 的 `ExceptionGroup` 正是为此设计的。

**`validate_batch(requests, validators)`**：对每个请求依次运行每个校验函数（校验函数接收请求，出错时抛出异常）。

- 全部通过时返回 `None`；
- 否则抛出一个 `ExceptionGroup`，消息是 `"{n} 个请求校验失败"`（`n` 是出错的请求数）；它的每个子异常对应一个出错的请求，也是一个 `ExceptionGroup`，消息是 `"request {i}"`（`i` 是请求下标），里面是这个请求的所有校验错误，顺序与 `validators` 一致。

**`count_errors(fn)`**：调用 `fn()`，用 `except*` 分类统计它抛出的错误，返回字典 `{"value": 值错误数, "type": 类型错误数}`。
其中 `ValueError` 和 `TypeError` 可能嵌套在多层 `ExceptionGroup` 里；其他类型的异常不要捕获，让它们继续抛出。`fn` 没有抛异常时返回两个 0。

```python
def need_prompt(r):
    if "prompt" not in r: raise ValueError("缺少 prompt")

def need_int_tokens(r):
    if not isinstance(r.get("max_tokens", 1), int): raise TypeError("max_tokens 必须是整数")

validate_batch([{"prompt": "hi"}, {"max_tokens": "8"}], [need_prompt, need_int_tokens])
# ExceptionGroup: 1 个请求校验失败 (1 sub-exception)
#   request 1: [ValueError('缺少 prompt'), TypeError('max_tokens 必须是整数')]

count_errors(lambda: validate_batch(...))   # {"value": 1, "type": 1}
```

<!-- 题解 -->
`validate_batch`：两层循环收集异常，`ExceptionGroup(f"request {i}", errors)` 包每个请求，外层再包一次。

`count_errors`：

```python
counts = {"value": 0, "type": 0}
try:
    fn()
except* ValueError as eg:
    counts["value"] += len(_leaves(eg))
except* TypeError as eg:
    counts["type"] += len(_leaves(eg))
```

`except*` 会把匹配的叶子异常"拆"出来，但保留原来的嵌套结构，所以要递归数叶子（或者用 `eg.exceptions` 展开）。
没有匹配的异常（比如 `KeyError`）会被自动重新抛出。
