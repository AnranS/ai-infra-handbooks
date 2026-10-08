# Threads, processes and the GIL

<p class="lead">Concurrency is where Python code most easily "works when written and breaks in production". This chapter starts with what the GIL actually restricts, then covers the correct use of threads, processes and <code>concurrent.futures</code>, and ends with a table for choosing between them.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is the GIL? Why does downloading files in several threads get faster while computing primes does not?
    2. Is `counter += 1` safe across threads?
    3. What do `ProcessPoolExecutor`'s task function and arguments have to satisfy?
    4. Why does multiprocessing code have to go inside `if __name__ == "__main__":`?
    5. How do `executor.map` and `submit` + `as_completed` differ?

??? success "Answers (try it yourself first, then expand)"
    1. The global interpreter lock: only one thread executes Python bytecode at a time. A thread waiting on I/O (the network, the disk) releases the GIL, so downloading in several threads overlaps; computing primes is pure Python CPU work, where the threads only take turns and gain nothing.
    2. It is not: it is several bytecodes, "read, add one, write back", and a thread may be switched out in the middle, losing an update. It needs a `threading.Lock`.
    3. Both the task function and the arguments have to be picklable (to be sent to a subprocess), so the function has to be defined at a module's top level and cannot be a lambda or a nested function.
    4. Under the spawn method (the default on Windows and macOS), the subprocess re-imports the main module; without that guard the subprocess would create subprocesses of its own, forever.
    5. `map` returns the results in the input's order, and an exception from a task is raised when that result is taken; `submit` returns a `Future`, and with `as_completed` the results are handled as they finish, which allows handling exceptions one at a time and showing progress.

<!-- comic ../assets/comics/gil.webp is in Chinese; put it back once the English version exists -->

## Telling the two kinds of task apart first {#先分清两类任务}

| | I/O-bound | CPU-bound |
| --- | --- | --- |
| Where the time goes | waiting on the network, the disk, the database | computing: parsing, compression, image processing, numerics |
| Examples | a crawler, calling an API, reading and writing files | hashing, training a model, transforming data at scale |
| The concurrency that fits | threads, asyncio | several processes, a C extension (NumPy and the like), free-threaded Python |

## The GIL: the global interpreter lock {#gil全局解释器锁}

CPython has a **global interpreter lock (GIL)**: only one thread executes Python bytecode at a time. So:

![Figure: the timeline of three situations under the GIL](../assets/figures/gil-timeline.svg){.aig-svg}

- **CPU-bound work**: several threads are no faster, since they only take turns and add the cost of switching.
- **I/O-bound work**: a thread waiting on I/O (`socket.recv`, `time.sleep`, reading a file) **releases the GIL** and the other threads can run. So threads remain very effective for I/O.
- C extensions like NumPy, zlib and hashlib release the GIL too while doing heavy computation.

!!! note "Free-threaded Python"
    From 3.13 there is an optional **free-threaded** build without the GIL, and from 3.14 it is officially supported (PEP 779), though it is **still not the default**. uv can install it: `uv python install 3.14t`, with the interpreter called `python3.14t`. It lets pure Python CPU-bound code run genuinely in parallel, at the cost of slightly slower single-threaded performance, and some C extensions have not adapted. Until it becomes the default, the rules below still apply.

## Threads, the basics {#线程基础}

```python
import threading
import time

def download(name, seconds, results):
    time.sleep(seconds)                       # simulating a network wait, which releases the GIL
    results[name] = f"{name} done"

results = {}
start = time.perf_counter()
threads = [
    threading.Thread(target=download, args=(f"file{i}", 0.2, results))
    for i in range(5)
]
for t in threads:
    t.start()
for t in threads:
    t.join()                                  # wait for every thread to finish
elapsed = time.perf_counter() - start

assert len(results) == 5
assert elapsed < 0.5                          # 5 tasks of 0.2 seconds run concurrently, about 0.2 seconds in all
```

Real code rarely creates a `Thread` directly and usually uses the thread pool below.

### Races and locks {#竞态条件与锁}

`counter += 1` is not atomic: it is three steps, "read, add one, write back", and a thread may be switched out in the middle, losing an update. **When several threads modify shared mutable state, a lock is required:**

```python
import threading

class Counter:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:                      # only one thread can be inside at a time
            self.value += 1

counter = Counter()

def work():
    for _ in range(10_000):
        counter.increment()

threads = [threading.Thread(target=work) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert counter.value == 80_000
```

!!! warning "Do not rely on the GIL for thread safety"
    With the GIL, unlocked code often "seems" to give the right answer, but that is luck. On free-threaded Python the races become far more frequent. **Shared mutable state takes a lock**, or better still: do not share state and pass messages through a queue.

The other synchronization primitives:

| Tool | Use |
| --- | --- |
| `Lock` | a mutex |
| `RLock` | a reentrant lock: the same thread can take it repeatedly |
| `Event` | one thread tells the others "something happened" (a stop signal, say) |
| `Semaphore(n)` | at most n threads inside at once (rate limiting) |
| `Condition` | waiting for a condition to hold, rather low level |
| `queue.Queue` | a thread-safe queue, **the preferred way for threads to communicate** |

### Producer-consumer: passing data through a queue {#生产者-消费者用队列传递数据}

```python
import queue
import threading

tasks: queue.Queue = queue.Queue()
results = []
results_lock = threading.Lock()

def worker():
    while True:
        item = tasks.get()
        if item is None:                      # the agreed stop signal
            tasks.task_done()
            break
        with results_lock:
            results.append(item * item)
        tasks.task_done()

workers = [threading.Thread(target=worker) for _ in range(3)]
for w in workers:
    w.start()
for n in range(10):
    tasks.put(n)
for _ in workers:
    tasks.put(None)                           # one stop signal per worker
tasks.join()                                  # wait for every item to be handled
for w in workers:
    w.join()

assert sorted(results) == [n * n for n in range(10)]
```

## `concurrent.futures`: thread pools and process pools {#concurrentfutures用线程池和进程池}

`concurrent.futures` offers one high-level interface for both. **It is enough for the great majority of concurrency needs**:

```python
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch(url):
    time.sleep(0.1)                           # simulating a network request
    if "bad" in url:
        raise ConnectionError(f"cannot reach {url}")
    return f"<html of {url}>"

urls = ["a.com", "b.com", "bad.com", "c.com"]

# the first way: map, whose results follow the input's order, with an exception raised when that result is taken
with ThreadPoolExecutor(max_workers=4) as pool:
    it = pool.map(fetch, ["a.com", "b.com"])
    assert list(it) == ["<html of a.com>", "<html of b.com>"]

# the second way: submit + as_completed, handling whichever finishes first and each exception on its own
ok, failed = {}, {}
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(fetch, url): url for url in urls}
    for fut in as_completed(futures):
        url = futures[fut]
        try:
            ok[url] = fut.result()
        except ConnectionError as e:
            failed[url] = str(e)

assert set(ok) == {"a.com", "b.com", "c.com"}
assert failed == {"bad.com": "cannot reach bad.com"}
```

- Leaving the `with` block waits for every task and shuts the pool down.
- An exception in a task **is not printed automatically** and sits in the `Future` until `result()` is called. A `submit` whose result is never taken swallows the exception silently.
- `fut.result(timeout=5)` sets how long to wait.
- The pool's size: a few dozen is fine for I/O, and the default is `min(32, the CPU count + 4)`.

## Several processes: real parallel computation {#多进程真正的并行计算}

Each process has its own interpreter and GIL, so several processes let CPU-bound work use several cores. `ProcessPoolExecutor`'s interface is identical to the thread pool's:

```python
import math
import time
from concurrent.futures import ProcessPoolExecutor

def count_primes(limit):
    count = 0
    for n in range(2, limit):
        if all(n % d for d in range(2, math.isqrt(n) + 1)):
            count += 1
    return count

def main():
    jobs = [60_000] * 4

    start = time.perf_counter()
    serial = [count_primes(n) for n in jobs]
    t_serial = time.perf_counter() - start

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(count_primes, jobs))
    t_parallel = time.perf_counter() - start

    assert serial == parallel == [6057] * 4
    print(f"serial {t_serial:.2f}s, 4 processes {t_parallel:.2f}s")

if __name__ == "__main__":
    main()
```

On a machine with 4 or more cores the parallel version is usually 2 to 4 times faster. A few things to watch:

1. **The `if __name__ == "__main__":` guard is required.** A subprocess re-imports the main module, and without the guard it would create a process pool of its own, forever.
2. **The task function and the arguments have to be picklable.** The function has to be defined at a module's top level (no lambdas, no nested functions) and the arguments and return values have to serialize.
3. **Communication between processes has a cost.** The arguments and results are serialized and sent across, so each task has to be large enough; do not submit millions of tiny tasks one by one (use `map`'s `chunksize`).
4. **The start method.** From 3.14 the default on Linux changed from `fork` to `forkserver`, which behaves closer to macOS's and Windows's `spawn`: the subprocess does not inherit the parent's global state.

!!! tip "3.14's new option: `InterpreterPoolExecutor`"
    3.14 adds `concurrent.futures.InterpreterPoolExecutor`: several subinterpreters in **one process**, each with its own GIL, running genuinely in parallel with less startup and communication cost than separate processes. It is still new and some C extensions do not support subinterpreters, so try it somewhere non-critical first.

## Choosing between them {#选型表}

| Case | Recommendation |
| --- | --- |
| a few concurrent I/O tasks (a few dozen API calls, downloading a batch of files) | `ThreadPoolExecutor` |
| heavy I/O concurrency (thousands of connections), long-lived connections, WebSockets | `asyncio` (the next chapter) |
| CPU-bound pure Python code | `ProcessPoolExecutor`; or try free-threaded Python |
| CPU-bound numerics | a vectorized library like NumPy or Polars (which releases the GIL and uses several cores inside) |
| calling a blocking function from an asyncio program | `asyncio.to_thread` |
| background scheduled work needing reliable retries and persistence | a task queue: Celery, RQ, Dramatiq |

!!! interview "How to explain it"
    On the GIL: only one thread executes Python bytecode at a time, but waiting on I/O and many C extensions (NumPy, PyTorch's operators) release it, so threads speed up downloading and not pure Python computation; `counter += 1` is several bytecodes, "read, modify, write", and is not atomic, so shared state takes a lock or moves to messages through a `queue.Queue`. CPU-bound work takes several processes: the task function and arguments have to be picklable (the function defined at a module's top level) and the startup code goes inside `if __name__ == "__main__":` (spawn re-imports the main module). That is exactly why an inference framework splits tokenizing, scheduling and GPU execution into separate processes passing messages over ZMQ (see SGLang's multi-process structure).

## Exercises {#练习}

**1. Fetching concurrently with a rate limit.** "Fetch" 20 URLs concurrently with a `ThreadPoolExecutor` (simulated with `time.sleep(0.05)`), allowing at most 5 requests in flight, and verify it by measuring the peak concurrency.

??? success "Answer"
    ```python
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    active = 0
    peak = 0
    lock = threading.Lock()

    def fetch(url):
        global active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.05)
            return url.upper()
        finally:
            with lock:
                active -= 1

    urls = [f"site{i}.com" for i in range(20)]
    with ThreadPoolExecutor(max_workers=5) as pool:     # the pool's size is the concurrency limit itself
        pages = list(pool.map(fetch, urls))

    assert pages[0] == "SITE0.COM" and len(pages) == 20
    assert peak <= 5
    ```

    When several thread pools share one downstream service and a global limit is wanted, use one shared `threading.Semaphore(5)` with a `with sem:` inside `fetch`.

**2. Thread-safe single initialization.** Rework the `@once` from [the decorators chapter](../core/decorators.md)'s exercises so it runs once even across threads. Verify it by calling it from 16 threads at once.

??? success "Answer"
    ```python
    import functools
    import threading
    import time

    def once(func):
        lock = threading.Lock()
        done = False
        result = None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal done, result
            if done:                         # the fast path: no lock once the initialization is done
                return result
            with lock:
                if not done:                 # double-checked: confirm again after taking the lock
                    result = func(*args, **kwargs)
                    done = True
            return result
        return wrapper

    calls = []

    @once
    def init():
        time.sleep(0.05)                     # widen the race window
        calls.append(1)
        return "ready"

    threads = [threading.Thread(target=init) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [1] and init() == "ready"
    ```

## Summary {#小结}

- [x] The GIL lets only one thread execute Python bytecode at a time, but it is released while waiting on I/O.
- [x] I/O-bound work takes threads or asyncio; CPU-bound work takes several processes or a vectorized library.
- [x] Shared mutable state requires a lock; passing messages through a `queue.Queue` is better still.
- [x] Prefer `concurrent.futures`; remember to call `result()`, or exceptions are swallowed.
- [x] Multiprocessing code goes inside `if __name__ == "__main__":` with the task function at a module's top level.
