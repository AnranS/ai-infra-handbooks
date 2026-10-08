# asyncio

<p class="lead">asyncio handles thousands of I/O tasks at once with one thread and one event loop. Its rules are few and every one of them matters: when to <code>await</code>, how to run tasks concurrently, how to handle timeouts and cancellation, and what must never be done inside a coroutine.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What does calling an `async def` function give you? Has the body run?
    2. How do `await coro()` and `asyncio.create_task(coro())` differ?
    3. What makes `TaskGroup` better than `gather`?
    4. What happens when a coroutine calls `time.sleep(1)`?
    5. How do you limit the number of requests in flight? How do you set one timeout over a group of operations?

??? success "Answers (try it yourself first, then expand)"
    1. A coroutine object, with not one line of the body run; it starts only when it is `await`ed or wrapped into a task and handed to the event loop.
    2. `await coro()` runs it immediately and waits for it, suspending the current coroutine meanwhile, so several `await`s in a row run one after another; `create_task` wraps it into a task for the event loop and returns at once, so the task runs in the background alongside the current coroutine and is `await`ed later for its result.
    3. Structured concurrency: leaving the `async with` block guarantees every task has finished; when one task fails the rest are cancelled and every exception is collected into an `ExceptionGroup`. `gather` by default does not cancel the others on an exception, which easily leaves background tasks running loose.
    4. It blocks the whole event loop thread, and every other coroutine is stopped for that second. Use `await asyncio.sleep(1)`; where a blocking function has to be called, put it in a thread with `asyncio.to_thread`.
    5. Concurrency: an `asyncio.Semaphore` (an `async with sem:` around the request), or a fixed number of workers taking tasks from a `Queue`; one overall timeout: wrap the whole group in `async with asyncio.timeout(seconds):` (which cancels the tasks inside on expiry).

## The core concepts {#核心概念}

- **a coroutine function**: a function defined with `async def`.
- **a coroutine object**: what calling a coroutine function gives. **Not one line of the body has run at that point.**
- **`await`**: runs a coroutine (or another awaitable) to completion and takes its result. While waiting on I/O, `await` hands control back to the event loop so other tasks can run.
- **the event loop**: the scheduler, which keeps picking tasks that "can continue" and running them.
- **a Task**: a coroutine scheduled by the event loop. `create_task` wraps a coroutine into a Task, which **runs concurrently in the background**.

```python
import asyncio

async def greet(name, delay):
    await asyncio.sleep(delay)          # a non-blocking wait: control is handed back
    return f"hello {name}"

coro = greet("amy", 0)                  # this only creates a coroutine object
assert asyncio.iscoroutine(coro)
assert asyncio.run(coro) == "hello amy" # asyncio.run: creates the event loop and runs to completion
```

`asyncio.run()` is the program's entry point and is usually called once.

![Figure: what the event loop is doing](../assets/figures/event-loop.svg){.aig-svg}

## Sequential and concurrent {#顺序执行与并发执行}

**`await` alone is sequential**, which is the commonest misunderstanding:

```python
import asyncio
import time

async def fetch(name, delay=0.1):
    await asyncio.sleep(delay)
    return name

async def sequential():
    return [await fetch("a"), await fetch("b"), await fetch("c")]   # one finishes before the next begins

async def concurrent():
    async with asyncio.TaskGroup() as tg:                             # 3.11+
        tasks = [tg.create_task(fetch(n)) for n in "abc"]
    return [t.result() for t in tasks]                                # everything has finished on leaving the async with

for fn, expected_time in [(sequential, 0.3), (concurrent, 0.1)]:
    start = time.perf_counter()
    assert asyncio.run(fn()) == ["a", "b", "c"]
    elapsed = time.perf_counter() - start
    assert abs(elapsed - expected_time) < 0.08, (fn.__name__, elapsed)
```

For concurrency, the coroutines first have to become **tasks** (`create_task`) so that they are all "on the road" at once, and only then awaited.

### `TaskGroup` and `gather` {#taskgroup-与-gather}

Both run a group of coroutines concurrently, and they differ in **what happens on failure**:

| | `asyncio.TaskGroup` <span class="since">3.11+</span> | `asyncio.gather` |
| --- | --- | --- |
| When one task fails | **cancels** the group's other tasks automatically and raises every exception in an `ExceptionGroup` | raises the first exception at you by default, while **the other tasks keep running in the background** |
| Getting the results | from each task's `result()` | returns a list of results directly, in the arguments' order |
| Recommended | **the first choice for new code** | still useful when `return_exceptions=True` is wanted to collect every outcome |

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
    except* ValueError as eg:              # an except* block cannot return, so record it first
        errors = [str(e) for e in eg.exceptions]
    return errors

async def with_gather():
    return await asyncio.gather(*(job(n) for n in range(1, 5)), return_exceptions=True)

assert asyncio.run(with_taskgroup()) == ["job 2 failed"]
results = asyncio.run(with_gather())
assert results[0] == 1 and isinstance(results[1], ValueError) and results[3] == 4
```

How `except*` is used is in [exceptions and context managers](../core/errors-context.md#同时处理多个异常exceptiongroup).

### Handling whichever finishes first: `as_completed` {#谁先完成先处理as_completed}

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

## Timeouts {#超时}

```python
import asyncio

async def slow():
    await asyncio.sleep(10)

async def main():
    try:
        async with asyncio.timeout(0.05):        # 3.11+: one timeout over a whole stretch of code
            await slow()
    except TimeoutError:
        return "timed out"

assert asyncio.run(main()) == "timed out"
```

`asyncio.timeout()` can wrap several `await`s, giving one **overall** timeout to a group of operations; a single wait can also use `await asyncio.wait_for(coro, timeout)`. On expiry, the coroutine running inside is **cancelled**.

## Cancellation {#取消}

Cancelling a task raises `asyncio.CancelledError` at the `await` where it is suspended:

```python
import asyncio

cleaned_up = []

async def worker():
    try:
        await asyncio.sleep(10)
    finally:
        cleaned_up.append(True)                  # the cleanup still runs on cancellation

async def main():
    task = asyncio.create_task(worker())
    await asyncio.sleep(0.01)                    # let the worker reach its sleep first
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return "cancelled"

assert asyncio.run(main()) == "cancelled" and cleaned_up == [True]
```

!!! warning "Do not swallow `CancelledError`"
    `CancelledError` inherits from `BaseException`, so `except Exception` does not catch it, which is deliberate. If you do catch it explicitly to clean up, **re-raise it without fail**, or timeouts and `TaskGroup`'s cancellation both stop working.

## Limiting concurrency: `Semaphore` {#限制并发数semaphore}

Ten thousand simultaneous requests will knock a service over or trip its rate limiter. A semaphore controls how many run at once:

```python
import asyncio

active = 0
peak = 0

async def fetch(sem, url):
    global active, peak
    async with sem:                              # at most limit tasks inside at a time
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

Note that the `active += 1` here needs no lock: asyncio is single-threaded and **the code between two `await`s cannot be interrupted**.

## Producer-consumer: `asyncio.Queue` {#生产者-消费者asyncioqueue}

A fixed number of workers taking tasks from a queue is the classic structure for a large batch of work, limiting the concurrency naturally and making backpressure easy (a full queue makes the producer wait):

```python
import asyncio

async def producer(queue, n):
    for i in range(n):
        await queue.put(i)                       # waits here while the queue is full

async def consumer(queue, results):
    while True:
        item = await queue.get()
        try:
            await asyncio.sleep(0)               # simulating the work
            results.append(item * 10)
        finally:
            queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=5)
    results = []
    workers = [asyncio.create_task(consumer(queue, results)) for _ in range(3)]
    await producer(queue, 20)
    await queue.join()                           # wait for every item to be handled
    for w in workers:
        w.cancel()                               # the workers loop forever, so cancel them once the work is done
    await asyncio.gather(*workers, return_exceptions=True)
    return sorted(results)

assert asyncio.run(main()) == [i * 10 for i in range(20)]
```

## Calling blocking code from asynchronous code {#在异步代码里调用阻塞代码}

**The event loop is single-threaded.** Any blocking call (`time.sleep`, `requests.get`, a synchronous database driver, heavy computation) stops **the whole program**, and no other task can run.

```python
import asyncio
import time

def blocking_io(x):
    time.sleep(0.1)                              # a synchronous library function
    return x * 2

async def main():
    start = time.perf_counter()
    results = await asyncio.gather(*(asyncio.to_thread(blocking_io, i) for i in range(5)))
    return results, time.perf_counter() - start

results, elapsed = asyncio.run(main())
assert results == [0, 2, 4, 6, 8] and elapsed < 0.3      # run concurrently in the thread pool
```

- A blocking I/O function: `await asyncio.to_thread(func, *args)`.
- A CPU-bound function: `loop.run_in_executor(process_pool, func, *args)` to put it in a process pool.
- The best answer is a natively asynchronous library: `httpx` or `aiohttp` for HTTP, `asyncpg` for PostgreSQL, `redis.asyncio` for Redis.

## Asynchronous context managers and iterators {#异步上下文管理器与异步迭代器}

```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def connection(name):
    await asyncio.sleep(0)                       # simulating opening a connection
    try:
        yield f"conn-{name}"
    finally:
        await asyncio.sleep(0)                   # simulating closing a connection

async def ticker(n):                             # an asynchronous generator
    for i in range(n):
        await asyncio.sleep(0)
        yield i

async def main():
    async with connection("db") as conn:
        values = [i async for i in ticker(3)]    # an asynchronous comprehension
    return conn, values

assert asyncio.run(main()) == ("conn-db", [0, 1, 2])
```

`async with` calls `__aenter__`/`__aexit__` and `async for` calls `__aiter__`/`__anext__`, and both are only usable inside an `async def`.

## The common traps {#常见坑}

- **Forgetting the `await`.** `fetch()` without it only creates a coroutine object and throws it away, with a later `RuntimeWarning: coroutine 'fetch' was never awaited`.
- **A "fire and forget" task being collected.** When nothing holds a reference to `asyncio.create_task(x())`'s return value, the task may be garbage collected mid-run. Either keep the reference (in a set, removed when it finishes) or use a `TaskGroup`.
- **A blocking call inside a coroutine.** See the section above. Debug mode (`asyncio.run(main(), debug=True)` or the environment variable `PYTHONASYNCIODEBUG=1`) reports callbacks that ran too long.
- **Nesting `asyncio.run` inside synchronous code.** With an event loop already running (inside Jupyter, say), `asyncio.run` cannot be called again; just `await`.
- **Thinking asyncio speeds up computation.** It does not, and only helps with waiting on I/O.

!!! tip "3.14's debugging tools"
    3.14 adds `python -m asyncio ps <PID>` and `python -m asyncio pstree <PID>`, which show the state and the call relationships of every asyncio task in a running Python process, which is very useful when "the program has hung".

!!! interview "How to explain it"
    On asyncio: calling an `async def` only gives a coroutine object and `await` runs it; several `await`s in a row are sequential, and concurrency takes `create_task` / `TaskGroup` / `gather`; `TaskGroup` cancels the other tasks when one fails and collects the exceptions, which is safer than `gather`; timeouts use `asyncio.timeout` and rate limiting uses a `Semaphore` or a fixed number of workers with a `Queue`; cancellation raises `CancelledError` at the `await` and must not be swallowed, with the cleanup in a `finally`. A blocking call in a coroutine (`time.sleep`, a synchronous HTTP request, heavy computation) stalls the whole event loop and belongs in `asyncio.to_thread` or another process. An inference service's API layer (FastAPI, streaming output) is built on exactly this.

## Exercises {#练习}

**1. A batch of requests with timeouts and retries.** Write `fetch_all(urls, limit=5, timeout=0.05, retries=2)`: request every URL concurrently (simulated with `asyncio.sleep`, where a URL containing `slow` takes 1 second and one containing `flaky` raises `ConnectionError` the first time), with a timeout per request, retrying on failure, finally returning `{url: the result or an error description}`, where one URL's failure does not affect the others.

??? success "Answer"
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

    `fetch_one` swallows the expected exceptions itself and returns an error description, so the `TaskGroup` does not cancel the other tasks when one URL fails. That is the usual form of "one failure does not affect the whole".

**2. An asynchronous pipeline.** Chain three stages through two queues: `produce` generates 1 to 20 → 2 `square` workers compute the squares → 1 `collect` sums them. Once all the data is processed, the program has to exit cleanly (with no task left behind).

??? success "Answer"
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
                await in_q.put(DONE)              # so the other square workers see the stop signal too
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
            await q1.put(DONE)                    # production finished, send the stop signal downstream

        async def square_then_signal():
            await square(q1, q2)
            await q2.put(DONE)                    # each worker notifies downstream once as it finishes

        async with asyncio.TaskGroup() as tg:
            tg.create_task(produce_then_signal())
            for _ in range(workers):
                tg.create_task(square_then_signal())
            total_task = tg.create_task(collect(q2, workers))
        return total_task.result()

    assert asyncio.run(pipeline()) == sum(i * i for i in range(1, 21))
    ```

    The sentinel object `DONE` means "no more data" and is passed down the pipeline stage by stage: the producer sends one `DONE` when it finishes; each `square` worker that sees `DONE` puts it back in the queue (so its peers see it too), exits, and sends one `DONE` downstream; `collect` finishes once it has seen as many `DONE`s as there are workers. Every task returns naturally and the `TaskGroup` exits cleanly.

## Summary {#小结}

- [x] Calling a coroutine function only creates a coroutine object; `await` runs it.
- [x] Several `await`s written together are sequential; concurrency takes `TaskGroup` / `create_task` / `gather`.
- [x] New code manages a group of tasks with `TaskGroup` and sets timeouts with `asyncio.timeout`.
- [x] Do not swallow `CancelledError`; put the cleanup in a `finally`.
- [x] Limit concurrency with a `Semaphore` or a fixed number of workers with a `Queue`.
- [x] A coroutine must never call a blocking function; use `asyncio.to_thread` where one is unavoidable.
