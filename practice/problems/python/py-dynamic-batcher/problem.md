---
title: 动态批处理器
chapter: concurrency/asyncio.md
difficulty: 困难
tags: [asyncio, Future, 批处理]
---
推理服务把同时到达的请求攒成一批再送进模型，以提高 GPU 利用率。用 asyncio 实现 `DynamicBatcher`：

```python
async def model(items: list) -> list:      # 一次处理一批，返回同样长度的结果
    ...

batcher = DynamicBatcher(model, max_batch_size=8, max_wait=0.01)
result = await batcher.submit(x)            # 在 async 函数里调用，等到自己的结果
await batcher.close()
```

规则：

- 一批从**第一个请求到达**时开始攒：攒够 `max_batch_size` 个立刻处理；或者距离这批第一个请求到达已经过了 `max_wait` 秒，就把当前攒到的处理掉；
- 同一时刻只处理一批（上一批 `await model(...)` 结束后才开始下一批）；处理期间到达的请求进入下一批；
- `model` 返回的列表按顺序对应每个请求；如果 `model` 抛出异常，这一批的每个 `submit` 都抛出这个异常，但批处理器继续工作；
- `batch_sizes` 属性记录每一批的大小（列表）；
- `close()`：处理完已经提交的请求后停止；之后再 `submit` 抛出 `RuntimeError`。

<!-- 题解 -->
`submit` 创建一个 `Future`，把 `(item, future)` 放进 `asyncio.Queue`，然后 `await future`。

后台任务（第一次 `submit` 时用 `asyncio.ensure_future` 启动）循环：

1. `await queue.get()` 等到这批的第一个请求，记下截止时间 `deadline = loop.time() + max_wait`；
2. 在 `len(batch) < max_batch_size` 时用 `asyncio.wait_for(queue.get(), deadline - now)` 继续取，超时就停；
3. `await model(items)`，逐个 `future.set_result(...)`；异常时逐个 `set_exception`。

关闭：往队列里放一个哨兵对象，后台任务取到哨兵就先处理完当前批再退出；`close()` 等后台任务结束。
