---
title: 限制并发数的 gather
chapter: concurrency/asyncio.md
difficulty: 中等
tags: [asyncio, Semaphore, 取消]
---
压测推理服务时，要同时发很多请求，但并发数不能超过上限。实现：

```python
async def gather_limited(factories, limit):
    ...
```

- `factories` 是一个列表，每个元素是**无参数的 async 函数**（调用它才得到协程，这样还没轮到的请求不会被提前创建）；
- 同一时刻最多有 `limit` 个在运行；一个结束就立刻启动下一个；
- 返回结果列表，**顺序和 `factories` 一致**（不是完成顺序）；
- 任何一个抛出异常时：取消所有还在运行的任务、不再启动新的，然后把这个异常抛出去；
- `limit < 1` 时抛出 `ValueError`。

```python
async def fetch(i):
    await asyncio.sleep(0.01)
    return i * i

await gather_limited([lambda i=i: fetch(i) for i in range(10)], limit=3)   # [0, 1, 4, ..., 81]
```

<!-- 题解 -->
两种常见写法：

1. `asyncio.Semaphore(limit)` + 一次性创建所有任务：每个任务先 `async with sem:` 再调用工厂。简单，但任务对象一次全部创建；
2. 固定 `limit` 个 worker 从共享的下标迭代器里取活：`next(it)` 取下标，把结果写进 `results[i]`。

出错时的处理：`asyncio.gather(*tasks)` 默认不会取消其他任务，所以要在 `except` 里对所有未完成的任务调用 `cancel()`，
再 `await asyncio.gather(*tasks, return_exceptions=True)` 等它们真正结束。Python 3.11 起也可以用 `asyncio.TaskGroup`，它会自动完成这些。
