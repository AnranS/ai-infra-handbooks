---
title: 重试与缓存装饰器
chapter: core/decorators.md
difficulty: 中等
tags: [装饰器, functools.wraps, 带参数的装饰器]
---
实现两个装饰器。

**`retry(times, exceptions=(Exception,), on_retry=None)`**：被装饰的函数抛出 `exceptions` 里的异常时自动重试，最多**总共调用** `times` 次；最后一次仍失败就把异常原样抛出。其他类型的异常不重试、直接抛出。每次重试前调用 `on_retry(attempt, exc)`（`attempt` 从 1 开始，表示第几次失败）。

**`memoize(maxsize=None)`**：按参数缓存返回值（参数都是可哈希的，支持关键字参数：`f(1, b=2)` 和 `f(1, 2)` 可以当作不同的调用）。
`maxsize` 不为 `None` 时最多缓存 `maxsize` 个结果，满了淘汰**最久没用到**的。被装饰的函数上还要有：

- `f.cache_info()`：返回 `(hits, misses, currsize)`；
- `f.cache_clear()`：清空缓存和计数。

两个装饰器都要保留原函数的 `__name__`、`__doc__`（用 `functools.wraps`）。

```python
@retry(3, exceptions=(ConnectionError,))
def fetch(): ...

@memoize(maxsize=128)
def fib(n): return n if n < 2 else fib(n - 1) + fib(n - 2)
```

<!-- 题解 -->
带参数的装饰器是"返回装饰器的函数"，一共三层：`retry(...)` → `decorator(fn)` → `wrapper(*args, **kwargs)`。

`memoize` 的键用 `(args, tuple(sorted(kwargs.items())))`；LRU 用 `OrderedDict`：命中时 `move_to_end`，插入后超出容量就 `popitem(last=False)`。
`cache_info`、`cache_clear` 直接作为属性挂在 `wrapper` 上。
