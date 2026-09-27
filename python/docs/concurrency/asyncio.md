# asyncio

<p class="lead">asyncio 用一个线程、一个事件循环，同时处理成千上万个 I/O 任务。它的规则不多，但每一条都很重要：什么时候用 <code>await</code>，怎么并发地运行任务，怎么处理超时和取消，以及绝对不能在协程里做什么。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 调用一个 `async def` 函数，得到的是什么？函数体执行了吗？
    2. `await coro()` 和 `asyncio.create_task(coro())` 有什么区别？
    3. `TaskGroup` 比 `gather` 好在哪？
    4. 在协程里调用 `time.sleep(1)` 会发生什么？
    5. 怎么限制同时进行的请求数？怎么给一组操作设置总超时？

## 核心概念

- **协程函数**：用 `async def` 定义的函数。
- **协程对象**：调用协程函数得到的对象。**此时函数体一行都没有执行**。
- **`await`**：运行一个协程（或其他可等待对象）直到它完成，并拿到结果。在等待 I/O 时，`await` 会把控制权交还给事件循环，让别的任务运行。
- **事件循环**：调度器，不断挑选"可以继续运行"的任务来执行。
- **Task**：被事件循环调度的协程。`create_task` 把协程包装成 Task，它会**在后台并发运行**。

```python
import asyncio

async def greet(name, delay):
    await asyncio.sleep(delay)          # 非阻塞的等待：让出控制权
    return f"hello {name}"

coro = greet("amy", 0)                  # 只是创建了协程对象
assert asyncio.iscoroutine(coro)
assert asyncio.run(coro) == "hello amy" # asyncio.run：创建事件循环，运行到结束
```

`asyncio.run()` 是程序的入口，一个程序通常只调用一次。

## 顺序执行与并发执行

**只写 `await` 是顺序执行的**，这是新手最常见的误解：

```python
import asyncio
import time

async def fetch(name, delay=0.1):
    await asyncio.sleep(delay)
    return name

async def sequential():
    return [await fetch("a"), await fetch("b"), await fetch("c")]   # 一个等完再下一个

async def concurrent():
    async with asyncio.TaskGroup() as tg:                             # 3.11+
        tasks = [tg.create_task(fetch(n)) for n in "abc"]
    return [t.result() for t in tasks]                                # 离开 async with 时全部已完成

for fn, expected_time in [(sequential, 0.3), (concurrent, 0.1)]:
    start = time.perf_counter()
    assert asyncio.run(fn()) == ["a", "b", "c"]
    elapsed = time.perf_counter() - start
    assert abs(elapsed - expected_time) < 0.08, (fn.__name__, elapsed)
```

要并发，必须先把协程变成**任务**（`create_task`），让它们同时"在路上"，然后再等它们。

### `TaskGroup` 与 `gather`

两者都能并发运行一组协程，区别在于**出错时的行为**：

| | `asyncio.TaskGroup` <span class="since">3.11+</span> | `asyncio.gather` |
| --- | --- | --- |
| 一个任务失败时 | **自动取消**同组其他任务，把所有异常打包成 `ExceptionGroup` 抛出 | 默认把第一个异常抛给你，**其他任务继续在后台跑** |
| 返回结果 | 从各个 task 的 `result()` 取 | 直接返回结果列表，顺序和参数一致 |
| 推荐程度 | **新代码首选** | 需要 `return_exceptions=True` 收集所有结果时仍然好用 |

```python
import asyncio

async def job(n):
    await asyncio.sleep(0.01 * n)
    if n == 2:
        raise ValueError(f"job {n} failed")
    return n

async def with_taskgroup():
    errors = []
    try:
        async with asyncio.TaskGroup() as tg:
            for n in range(1, 5):
                tg.create_task(job(n))
    except* ValueError as eg:              # except* 块里不能 return，先记下来
        errors = [str(e) for e in eg.exceptions]
    return errors

async def with_gather():
    return await asyncio.gather(*(job(n) for n in range(1, 5)), return_exceptions=True)

assert asyncio.run(with_taskgroup()) == ["job 2 failed"]
results = asyncio.run(with_gather())
assert results[0] == 1 and isinstance(results[1], ValueError) and results[3] == 4
```

`except*` 的用法见[异常与上下文管理器](../core/errors-context.md#同时处理多个异常exceptiongroup)。

### 谁先完成先处理：`as_completed`

```python
import asyncio

async def fetch(name, delay):
    await asyncio.sleep(delay)
    return name

async def main():
    order = []
    coros = [fetch("slow", 0.06), fetch("fast", 0.01), fetch("mid", 0.03)]
    for next_done in asyncio.as_completed(coros):
        order.append(await next_done)
    return order

assert asyncio.run(main()) == ["fast", "mid", "slow"]
```

## 超时

```python
import asyncio

async def slow():
    await asyncio.sleep(10)

async def main():
    try:
        async with asyncio.timeout(0.05):        # 3.11+：给一整段代码设置超时
            await slow()
    except TimeoutError:
        return "timed out"

assert asyncio.run(main()) == "timed out"
```

`asyncio.timeout()` 可以包住多个 `await`，给一组操作设置**总**超时；单个等待也可以用 `await asyncio.wait_for(coro, timeout)`。超时发生时，里面正在运行的协程会被**取消**。

## 取消

取消一个任务，会在它当前停着的 `await` 处抛出 `asyncio.CancelledError`：

```python
import asyncio

cleaned_up = []

async def worker():
    try:
        await asyncio.sleep(10)
    finally:
        cleaned_up.append(True)                  # 取消时清理代码照样执行

async def main():
    task = asyncio.create_task(worker())
    await asyncio.sleep(0.01)                    # 让 worker 先运行到 sleep
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return "cancelled"

assert asyncio.run(main()) == "cancelled" and cleaned_up == [True]
```

!!! warning "不要吞掉 `CancelledError`"
    `CancelledError` 继承自 `BaseException`，所以 `except Exception` 不会捕获它，这是有意设计的。如果你显式捕获了它做清理，**一定要重新抛出**，否则超时、`TaskGroup` 的取消都会失效。

## 限制并发数：`Semaphore`

同时向一个服务发起一万个请求会把它打垮，或者触发限流。用信号量控制同一时刻的并发数：

```python
import asyncio

active = 0
peak = 0

async def fetch(sem, url):
    global active, peak
    async with sem:                              # 同一时刻最多 limit 个任务进入
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return url

async def crawl(urls, limit=3):
    sem = asyncio.Semaphore(limit)
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(sem, u)) for u in urls]
    return [t.result() for t in tasks]

pages = asyncio.run(crawl([f"u{i}" for i in range(20)]))
assert len(pages) == 20 and peak == 3
```

注意这里的 `active += 1` 不需要锁：asyncio 是单线程的，**两个 `await` 之间的代码不会被打断**。

## 生产者-消费者：`asyncio.Queue`

固定数量的 worker 从队列取任务，是处理大批量任务的经典结构，天然限制了并发数，也方便做背压（队列满时生产者自动等待）：

```python
import asyncio

async def producer(queue, n):
    for i in range(n):
        await queue.put(i)                       # 队列满时在这里等待

async def consumer(queue, results):
    while True:
        item = await queue.get()
        try:
            await asyncio.sleep(0)               # 模拟处理
            results.append(item * 10)
        finally:
            queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=5)
    results = []
    workers = [asyncio.create_task(consumer(queue, results)) for _ in range(3)]
    await producer(queue, 20)
    await queue.join()                           # 等所有任务处理完
    for w in workers:
        w.cancel()                               # worker 是无限循环，处理完后取消它们
    await asyncio.gather(*workers, return_exceptions=True)
    return sorted(results)

assert asyncio.run(main()) == [i * 10 for i in range(20)]
```

## 在异步代码里调用阻塞代码

**事件循环是单线程的。** 任何阻塞调用（`time.sleep`、`requests.get`、同步的数据库驱动、大量 CPU 计算）都会让**整个程序**停住，所有其他任务都无法运行。

```python
import asyncio
import time

def blocking_io(x):
    time.sleep(0.1)                              # 同步的库函数
    return x * 2

async def main():
    start = time.perf_counter()
    results = await asyncio.gather(*(asyncio.to_thread(blocking_io, i) for i in range(5)))
    return results, time.perf_counter() - start

results, elapsed = asyncio.run(main())
assert results == [0, 2, 4, 6, 8] and elapsed < 0.3      # 在线程池里并发执行
```

- 阻塞的 I/O 函数：`await asyncio.to_thread(func, *args)`。
- CPU 密集型函数：用 `loop.run_in_executor(process_pool, func, *args)` 放进进程池。
- 最好的办法是用原生的异步库：HTTP 用 `httpx` 或 `aiohttp`，PostgreSQL 用 `asyncpg`，Redis 用 `redis.asyncio`。

## 异步上下文管理器与异步迭代器

```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def connection(name):
    await asyncio.sleep(0)                       # 模拟建立连接
    try:
        yield f"conn-{name}"
    finally:
        await asyncio.sleep(0)                   # 模拟关闭连接

async def ticker(n):                             # 异步生成器
    for i in range(n):
        await asyncio.sleep(0)
        yield i

async def main():
    async with connection("db") as conn:
        values = [i async for i in ticker(3)]    # 异步推导式
    return conn, values

assert asyncio.run(main()) == ("conn-db", [0, 1, 2])
```

`async with` 调用 `__aenter__`/`__aexit__`，`async for` 调用 `__aiter__`/`__anext__`，只能在 `async def` 里使用。

## 常见坑

- **忘了 `await`。** `fetch()` 不加 `await` 只会创建一个协程对象然后丢掉，并在之后警告 `RuntimeWarning: coroutine 'fetch' was never awaited`。
- **"发射后不管"的任务被回收。** `asyncio.create_task(x())` 的返回值如果没人引用，任务可能在执行途中被垃圾回收。要么保存引用（放进集合，完成后移除），要么用 `TaskGroup`。
- **在协程里用了阻塞调用。** 见上一节。开启调试模式（`asyncio.run(main(), debug=True)` 或环境变量 `PYTHONASYNCIODEBUG=1`）会报告执行时间过长的回调。
- **在同步代码里调用 `asyncio.run` 嵌套。** 已经有事件循环在运行时（比如在 Jupyter 里），不能再调用 `asyncio.run`，直接 `await` 即可。
- **以为 asyncio 能加速 CPU 计算。** 不能，它只对 I/O 等待有效。

!!! tip "3.14 的调试利器"
    3.14 新增了 `python -m asyncio ps <PID>` 和 `python -m asyncio pstree <PID>`，可以查看一个正在运行的 Python 进程里所有 asyncio 任务的状态和调用关系，排查"程序卡住了"时非常有用。

## 练习

**1. 带超时和重试的批量请求。** 写 `fetch_all(urls, limit=5, timeout=0.05, retries=2)`：并发请求所有 URL（用 `asyncio.sleep` 模拟，URL 里含 `slow` 的耗时 1 秒，含 `flaky` 的第一次会抛 `ConnectionError`），每个请求有单独的超时，失败时重试，最终返回 `{url: 结果或错误描述}`，一个 URL 失败不影响其他 URL。

??? success "参考答案"
    ```python
    import asyncio

    attempts: dict[str, int] = {}

    async def fake_request(url):
        attempts[url] = attempts.get(url, 0) + 1
        if "slow" in url:
            await asyncio.sleep(1)
        if "flaky" in url and attempts[url] == 1:
            raise ConnectionError("reset by peer")
        await asyncio.sleep(0.01)
        return f"200 {url}"

    async def fetch_one(sem, url, timeout, retries):
        last_error = None
        for _ in range(retries + 1):
            try:
                async with sem, asyncio.timeout(timeout):
                    return await fake_request(url)
            except (TimeoutError, ConnectionError) as e:
                last_error = e
        return f"failed: {type(last_error).__name__}"

    async def fetch_all(urls, limit=5, timeout=0.05, retries=2):
        sem = asyncio.Semaphore(limit)
        async with asyncio.TaskGroup() as tg:
            tasks = {url: tg.create_task(fetch_one(sem, url, timeout, retries)) for url in urls}
        return {url: t.result() for url, t in tasks.items()}

    result = asyncio.run(fetch_all(["a", "flaky-b", "slow-c"]))
    assert result == {"a": "200 a", "flaky-b": "200 flaky-b", "slow-c": "failed: TimeoutError"}
    assert attempts == {"a": 1, "flaky-b": 2, "slow-c": 3}
    ```

    `fetch_one` 自己吞掉了预期内的异常并返回错误描述，所以 `TaskGroup` 不会因为一个 URL 失败而取消其他任务。这是"单个失败不影响整体"的常用写法。

**2. 异步流水线。** 用两个队列串起三个阶段：`produce` 生成 1 到 20 → 2 个 `square` worker 计算平方 → 1 个 `collect` 汇总求和。所有数据处理完后，程序要干净地退出（没有残留的任务）。

??? success "参考答案"
    ```python
    import asyncio

    DONE = object()

    async def produce(out_q, n):
        for i in range(1, n + 1):
            await out_q.put(i)

    async def square(in_q, out_q):
        while True:
            item = await in_q.get()
            if item is DONE:
                await in_q.put(DONE)              # 让其他 square worker 也能看到结束信号
                return
            await out_q.put(item * item)

    async def collect(in_q, expected_workers):
        total, finished = 0, 0
        while finished < expected_workers:
            item = await in_q.get()
            if item is DONE:
                finished += 1
            else:
                total += item
        return total

    async def pipeline(n=20, workers=2):
        q1, q2 = asyncio.Queue(maxsize=4), asyncio.Queue(maxsize=4)

        async def produce_then_signal():
            await produce(q1, n)
            await q1.put(DONE)                    # 生产完毕，向下游发结束信号

        async def square_then_signal():
            await square(q1, q2)
            await q2.put(DONE)                    # 每个 worker 结束时通知下游一次

        async with asyncio.TaskGroup() as tg:
            tg.create_task(produce_then_signal())
            for _ in range(workers):
                tg.create_task(square_then_signal())
            total_task = tg.create_task(collect(q2, workers))
        return total_task.result()

    assert asyncio.run(pipeline()) == sum(i * i for i in range(1, 21))
    ```

    用哨兵对象 `DONE` 表示"没有更多数据了"，沿着流水线逐级传递：生产者结束后发一个 `DONE`；每个 `square` worker 看到 `DONE` 后把它放回队列（让同伴也能看到）并退出，同时向下游发一个 `DONE`；`collect` 收齐 worker 数量个 `DONE` 后结束。所有任务都自然返回，`TaskGroup` 干净退出。

## 小结

- [x] 调用协程函数只创建协程对象；`await` 才会运行它。
- [x] 多个 `await` 写在一起是顺序执行；并发要用 `TaskGroup` / `create_task` / `gather`。
- [x] 新代码用 `TaskGroup` 管理一组任务，用 `asyncio.timeout` 设置超时。
- [x] 不要吞掉 `CancelledError`；清理代码写在 `finally` 里。
- [x] 用 `Semaphore` 或固定数量的 worker + `Queue` 限制并发。
- [x] 协程里绝不能调用阻塞函数；必要时用 `asyncio.to_thread`。
