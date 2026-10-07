# A lock-free queue and a thread pool

<p class="lead">With atomics and memory order in hand, the two concurrent components an inference system uses most can be written: a single-producer single-consumer lock-free ring queue (passing data between pipeline stages with no lock and no system call), and a thread pool (spreading the CPU's parallel work, tokenizing, loading weight shards, forming batches, over a fixed few threads). This chapter writes each one and verifies it under ThreadSanitizer.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why can a single-producer single-consumer (SPSC) queue do without a lock? Who writes each of its read and write pointers?
    2. Why is a ring queue's capacity usually a power of two?
    3. How does a thread pool's `submit` return a task's result and its exceptions to the caller?
    4. What happens to the tasks still in the queue when the thread pool is destroyed?
    5. What can go wrong when a task in a thread pool waits for another task submitted to **the same pool**?

??? success "Answers (try it yourself first, then expand)"
    1. Only one thread writes each pointer: the producer writes only `tail` (how far it has written) and the consumer only `head` (how far it has read), so no two threads modify the same variable. The producer writes the element and then updates `tail` with a release, the consumer reads `tail` with an acquire and then the element, and the handover is safe; the other direction is the same.
    2. An index modulo the capacity becomes a bitwise and (`index & (Cap - 1)`), which is faster than a division; and with the pointers increasing forever and taken modulo only when indexing the array, "full" and "empty" are still distinguishable.
    3. Wrap the task in a `std::packaged_task` and return its `std::future`: when a worker runs the task, the return value or the exception it throws goes into the shared state, and the caller's `future.get()` returns the result or rethrows the exception.
    4. The policy has to be explicit: usually set the stop flag, wake every thread, and let them finish the remaining tasks before exiting (or discard them), then `join` every thread. A thread must never touch the queue after it has been destroyed.
    5. Thread pool starvation deadlock: if every worker waits for a task still sitting in the queue, no thread is left to run them and they wait forever. Do not block on a task of the same pool from inside a task; use a separate pool or an asynchronous continuation instead.

## A single-producer single-consumer ring queue {#单生产者单消费者环形队列}

![Figure: a ring buffer - the producer writes head and the consumer reads tail](../assets/figures/ring-buffer.svg){.aig-svg}

Many of an inference service's pipeline queues naturally have one writer and one reader: the scheduler thread hands sampled tokens to the detokenizer thread, the network thread hands requests to the tokenizer thread, and the thread polling GPU completion events hands notifications to the scheduler. Such a queue can be **entirely lock-free**:

- a fixed-size array, a `tail` saying how far it has been written (written only by the producer) and a `head` saying how far it has been read (written only by the consumer);
- the producer writes the element and updates `tail` with a **release**; the consumer reads `tail` with an **acquire** and can then read the element safely, exactly the previous chapter's "publish" pattern; the other direction is the same, with the consumer updating `head` with a release to tell the producer that the slot may be reused;
- `head` and `tail` only ever increase and are mapped to an array index by `index & (capacity - 1)` (hence a power-of-two capacity, turning the modulo into a bitwise and), and `tail - head` is the current element count.

Two more optimizations: put `head` and `tail` in different cache lines (or the producer and the consumer falsely share); and have each **cache** a copy of the other's pointer, reading the other's atomic only when it "looks full / empty", which cuts the cache line traffic between cores.

```cpp title="spsc_queue.hpp"
#pragma once
#include <array>
#include <atomic>
#include <cstddef>

template <class T, std::size_t Cap>
class SpscQueue {
  static_assert(Cap > 0 && (Cap & (Cap - 1)) == 0, "容量必须是 2 的幂");

 public:
  bool try_push(const T& v) {   // only the producer thread may call this
    const std::size_t t = prod_.tail.load(std::memory_order_relaxed);
    if (t - prod_.head_cache == Cap) {                                 // it looks full: refresh the consumer's progress
      prod_.head_cache = cons_.head.load(std::memory_order_acquire);
      if (t - prod_.head_cache == Cap) return false;
    }
    buf_[t & (Cap - 1)] = v;
    prod_.tail.store(t + 1, std::memory_order_release);                // write the data first, then publish
    return true;
  }
  bool try_pop(T& out) {        // only the consumer thread may call this
    const std::size_t h = cons_.head.load(std::memory_order_relaxed);
    if (h == cons_.tail_cache) {                                       // it looks empty: refresh the producer's progress
      cons_.tail_cache = prod_.tail.load(std::memory_order_acquire);
      if (h == cons_.tail_cache) return false;
    }
    out = buf_[h & (Cap - 1)];
    cons_.head.store(h + 1, std::memory_order_release);                // read, so this slot may be reused
    return true;
  }

 private:
  struct alignas(64) Producer {       // what the producer writes goes in one cache line
    std::atomic<std::size_t> tail{0};
    std::size_t head_cache = 0;
  };
  struct alignas(64) Consumer {       // what the consumer writes goes in another
    std::atomic<std::size_t> head{0};
    std::size_t tail_cache = 0;
  };
  Producer prod_;
  Consumer cons_;
  std::array<T, Cap> buf_{};
};
```

```cpp title="spsc_demo.cpp" sanitize="thread"
#include <cstdio>
#include <thread>

#include "spsc_queue.hpp"

int main() {
  SpscQueue<long, 1024> q;
  const long n = 200000;
  std::thread producer([&] {
    for (long i = 1; i <= n; ++i) {
      while (!q.try_push(i)) std::this_thread::yield();   // yield the CPU while it is full
    }
  });
  long received = 0, sum = 0, last = 0;
  bool in_order = true;
  while (received < n) {
    long v;
    if (!q.try_pop(v)) {
      std::this_thread::yield();
      continue;
    }
    in_order = in_order && v == last + 1;
    last = v;
    sum += v;
    ++received;
  }
  producer.join();
  std::printf("收到 %ld 个，%s，和 = %ld\n", received, in_order ? "顺序正确" : "顺序错乱", sum);
}
```

```text title="output"
收到 200000 个，顺序正确，和 = 20000100000
```

This queue has no lock and no system call, and an operation is usually one atomic load and store, a few nanoseconds. The cost is that **when it is empty the consumer has to decide how to wait**: spin forever (the lowest latency, at the cost of a whole core), `yield`, or spin briefly and fall back to a condition variable or an atomic wait. Latency-sensitive cases (the thread polling an RDMA completion queue, say) usually pin a core and spin on it.

!!! note "What about several producers or consumers?"
    A multi-producer multi-consumer lock-free queue (Dmitry Vyukov's bounded MPMC queue, say) claims slots with CAS and its correctness argument is far more involved.
    The practical order is: use the previous chapter's mutex + condition variable queue first; once profiling proves the lock is the bottleneck, see whether the structure can become several SPSC queues (one per producer); and only then consider an MPMC implementation from a mature library, rather than writing one yourself.

## A thread pool {#线程池}

![Figure: a thread pool - the task queue, the workers, the condition variable and the stop flag](../assets/figures/thread-pool.svg){.aig-svg}

Creating and destroying threads costs something (tens of microseconds), and with far more threads than cores, context switching eats a great deal of time. A thread pool creates a fixed number of workers in advance, and tasks submitted to a shared queue are taken by whichever worker is free.

The implementation's key points:

- the task queue uses the previous chapter's "mutex + condition variable";
- `submit` returns a `std::future`: the task is wrapped in a `std::packaged_task`, which puts both the return value **and the exceptions** into the shared state, so the caller's `get()` returns the result or rethrows the exception in the caller's thread;
- **run the task outside the lock**, or the tasks serialize;
- on destruction, set the "stop" flag and wake every thread first, let the workers finish the remaining tasks before exiting (discarding them is also a choice), and `join` last.

```cpp title="thread_pool.hpp"
#pragma once
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(unsigned n) {
    for (unsigned i = 0; i < n; ++i) workers_.emplace_back([this] { run(); });
  }
  ~ThreadPool() {
    {
      std::lock_guard lk(m_);
      stopping_ = true;
    }
    cv_.notify_all();
  }   // then workers_ is destroyed (a jthread joins automatically): the workers finish the remaining tasks and exit

  template <class F>
  auto submit(F f) -> std::future<std::invoke_result_t<F>> {
    using R = std::invoke_result_t<F>;
    // a packaged_task can only be moved while std::function requires copyability, hence the shared_ptr
    auto task = std::make_shared<std::packaged_task<R()>>(std::move(f));
    auto fut = task->get_future();
    {
      std::lock_guard lk(m_);
      if (stopping_) throw std::runtime_error("线程池已经关闭");
      tasks_.emplace_back([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

 private:
  void run() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock lk(m_);
        cv_.wait(lk, [&] { return stopping_ || !tasks_.empty(); });
        if (tasks_.empty()) return;   // stopped, and the queue is drained
        job = std::move(tasks_.front());
        tasks_.pop_front();
      }
      job();                          // run outside the lock
    }
  }

  std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;   // protected by m_
  bool stopping_ = false;                     // protected by m_
  std::vector<std::jthread> workers_;         // declared last and destroyed first: the members above are still alive at the join
};
```

The order in which the members are declared matters here: `workers_` has to be declared last so that it is destroyed **first**, which lets the workers use the lock, the condition variable and the queue safely before the `join` (recall "members are destroyed in reverse order of declaration" from [value semantics and RAII](../basics/value-raii.md)).

```cpp title="pool_demo.cpp" sanitize="thread"
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "thread_pool.hpp"

// cut [0, n) into ranges for the pool and wait for them all
template <class F>
void parallel_for(ThreadPool& pool, int n, int chunks, F f) {
  std::vector<std::future<void>> fs;
  for (int c = 0; c < chunks; ++c) {
    int lo = n * c / chunks, hi = n * (c + 1) / chunks;
    fs.push_back(pool.submit([=] {
      for (int i = lo; i < hi; ++i) f(i);
    }));
  }
  for (auto& fu : fs) fu.get();   // an exception from any range is rethrown here
}

int main() {
  ThreadPool pool(4);

  std::vector<std::future<long>> results;
  for (long i = 0; i < 100; ++i) results.push_back(pool.submit([i] { return i * i; }));
  long sum = 0;
  for (auto& r : results) sum += r.get();
  std::printf("平方和 = %ld\n", sum);

  auto bad = pool.submit([]() -> int { throw std::runtime_error("分词失败：非法的 UTF-8"); });
  try {
    bad.get();
  } catch (const std::exception& e) {
    std::printf("任务里的异常在 get() 时抛出：%s\n", e.what());
  }

  std::vector<int> shard_tokens(8);   // one task per shard writes its own slot
  parallel_for(pool, 8, 3, [&](int i) { shard_tokens[i] = (i + 1) * 1000; });
  std::printf("8 个分片共 %d 个 token\n", std::accumulate(shard_tokens.begin(), shard_tokens.end(), 0));
}
```

```text title="output"
平方和 = 328350
任务里的异常在 get() 时抛出：分词失败：非法的 UTF-8
8 个分片共 36000 个 token
```

Besides returning the result, `future::get()` establishes synchronization: the writes to `shard_tokens` inside the tasks are visible to the main thread once `get()` returns, so reading it is safe and TSan reports nothing.

### A few traps with thread pools {#线程池的几个坑}

- **waiting inside a task for another task of the same pool**: if every worker waits for a task that "is still in the queue with nobody to run it", they wait forever (thread pool starvation deadlock). When tasks depend on each other, either use different pools or have the waiter run queued tasks itself (what work stealing does);
- **blocking I/O in a compute pool**: a task reading the network occupies a thread and the compute tasks queue up behind it. Put I/O in its own pool or use asynchronous I/O;
- **oversubscription**: a pool of 32 threads whose tasks each call PyTorch or OpenMP, which open 32 threads of their own, gives 1024 threads fighting over 32 cores and throughput falls sharply. The usual fix is to limit the inner thread count (`torch.set_num_threads`, `OMP_NUM_THREADS`);
- **tasks that are too small**: each task costs a lock and a wakeup, and when the task itself takes a few microseconds the scheduling dominates. Split the work into chunks (the `parallel_for` above cuts 8 elements into 3 ranges), or use a **work-stealing** structure with a local queue per thread that steals from others when idle (TBB, Rust's rayon and Go's scheduler all do this).

!!! interview "Answering in an interview"
    On the lock-free queue and the thread pool: a single-producer single-consumer ring queue can be entirely lock-free, with the producer writing only `tail` and the consumer only `head`, moving `tail` with a release after writing the data and `head` with a release after reading it, each side reading the other with an acquire; the capacity is a power of two so the index is a bitwise and; the two pointers go in different cache lines and each caches the other's value to cut contention. For several producers and consumers, use a lock-based queue first and optimize once it is proved to be the bottleneck. The thread pool: a task queue with a lock and a condition variable, a `packaged_task` handing the return value and the exceptions to a `future`, tasks run outside the lock, and destruction setting the stop flag and waking every thread before the join. Waiting inside a task for another task of the same pool deadlocks through starvation.

## Exercises {#练习}

1. Give `ThreadPool` a `wait_idle()`: block until the queue is empty and no task is running. Use it to rewrite a program that "submits 1000 tasks, keeps no futures and waits for them all at the end". It has to be clean under TSan.

??? success "Answer"
    Keep a count of unfinished tasks: incremented on submission, decremented when a task finishes, and notifying the waiters when it reaches 0. The count and the queue are protected by the same lock, and a second condition variable notifies `wait_idle`:

    ```cpp title="pool_idle.cpp" sanitize="thread"
    #include <atomic>
    #include <condition_variable>
    #include <cstdio>
    #include <deque>
    #include <functional>
    #include <mutex>
    #include <thread>
    #include <vector>

    class Pool {
     public:
      explicit Pool(unsigned n) {
        for (unsigned i = 0; i < n; ++i) workers_.emplace_back([this] { run(); });
      }
      ~Pool() {
        {
          std::lock_guard lk(m_);
          stopping_ = true;
        }
        cv_.notify_all();
      }
      void post(std::function<void()> f) {
        {
          std::lock_guard lk(m_);
          tasks_.push_back(std::move(f));
          ++pending_;
        }
        cv_.notify_one();
      }
      void wait_idle() {
        std::unique_lock lk(m_);
        idle_cv_.wait(lk, [&] { return pending_ == 0; });
      }

     private:
      void run() {
        for (;;) {
          std::function<void()> job;
          {
            std::unique_lock lk(m_);
            cv_.wait(lk, [&] { return stopping_ || !tasks_.empty(); });
            if (tasks_.empty()) return;
            job = std::move(tasks_.front());
            tasks_.pop_front();
          }
          job();
          {
            std::lock_guard lk(m_);
            if (--pending_ == 0) idle_cv_.notify_all();
          }
        }
      }
      std::mutex m_;
      std::condition_variable cv_, idle_cv_;
      std::deque<std::function<void()>> tasks_;
      int pending_ = 0;   // the number of tasks submitted and not yet finished
      bool stopping_ = false;
      std::vector<std::jthread> workers_;
    };

    int main() {
      Pool pool(4);
      std::atomic<int> done{0};
      for (int i = 0; i < 1000; ++i) pool.post([&] { done.fetch_add(1, std::memory_order_relaxed); });
      pool.wait_idle();
      std::printf("wait_idle 返回时完成了 %d 个任务\n", done.load());
    }
    ```

    ```text title="output"
    wait_idle 返回时完成了 1000 个任务
    ```

    Note that `pending_` is decremented **after the task has run** and not when it is taken from the queue: otherwise `wait_idle` could return while the last few tasks are still running.

2. In `SpscQueue`, what happens if `tail`'s `store` in `try_push` is changed to `relaxed`? And if `alignas(64)` is removed from `Producer` and `Consumer`?

??? success "Answer"
    With `tail` relaxed, the order between writing the element and updating `tail` is no longer guaranteed: the consumer may see the new `tail` first and read a slot that is not written yet, a data race that TSan reports and that really does read stale data on ARM.
    Removing `alignas(64)` does not affect correctness, but the `tail` the producer writes often and the `head` the consumer writes often would very likely land in the same cache line, each operation invalidating the other's line and cutting throughput several times over (see [false sharing](../memory/layout.md#伪共享)).

## Summary {#小结}

- [x] A single-producer single-consumer queue can be entirely lock-free: the producer writes only `tail` and the consumer only `head`, publishing the data and freeing the slot through release / acquire.
- [x] The capacity is a power of two so the index is a bitwise and; the two ends' pointers go in different cache lines and each caches the other's pointer.
- [x] For several producers and consumers, use a lock-based queue first, and only once it is proved to be the bottleneck consider several SPSC queues or a mature implementation.
- [x] The thread pool: a task queue with a lock and a condition variable, a `packaged_task` returning the result and the exceptions through a `future`, tasks run outside the lock, and destruction stopping before joining.
- [x] Watch out for thread pool starvation deadlock, blocking I/O, oversubscription and tasks that are too small.
