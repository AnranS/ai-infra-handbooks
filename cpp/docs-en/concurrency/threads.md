# Threads, locks and condition variables

<p class="lead">An inference service is concurrent throughout: the tokenizer, the scheduler and the detokenizer each run in a thread or a process, a network thread takes requests, background threads load weights and report metrics, and the communication library has proxy threads of its own. C++ threads run genuinely in parallel with no GIL to fall back on, and any two threads reading and writing one variable without synchronization is undefined behaviour. This chapter covers the correct use of threads, mutexes and condition variables, and proves with ThreadSanitizer that the code has no data race.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is a data race? How does it differ from a race condition?
    2. Where does each of `std::lock_guard`, `std::unique_lock` and `std::scoped_lock` belong?
    3. Why must a condition variable's `wait` carry a predicate (or sit in a `while` loop)?
    4. What happens when two threads take two locks in different orders? How do you avoid it?
    5. What does `std::jthread` add over `std::thread`?

??? success "Answers (try it yourself first, then expand)"
    1. A data race: two threads access the same memory location, at least one of them writing, with no synchronization between them, which is undefined behaviour. A race condition is the broader logical problem: the result depends on the order in which the threads run, and can go wrong even when every access is atomic.
    2. `lock_guard`: the simplest scoped lock; `unique_lock`: can lock late, unlock partway and be moved, and a condition variable has to use it; `scoped_lock`: locks several mutexes at once with a deadlock-free algorithm.
    3. There are spurious wakeups (waking without a notify), and the condition may have been changed by another thread by the time it wakes; so the condition is rechecked on every wake and it goes back to waiting if it does not hold.
    4. It may deadlock: each holds one lock and waits for the other's. To avoid it, have every thread take the locks in a fixed order, or lock them together with `std::scoped_lock`.
    5. It `join`s automatically on destruction (a `std::thread` that was neither joined nor detached calls `terminate` outright), and it carries a `stop_token` for asking the thread to exit cooperatively.

## Threads and data races {#线程与数据竞争}

`std::thread` creates a thread and `join()` waits for it. If a thread object is destroyed without a `join()` (or `detach()`), the program calls `std::terminate` outright, which is why `std::jthread` is preferred from C++20: it requests a stop and `join()`s automatically on destruction (the last section of this chapter).

**A data race**: two threads access the same memory location, at least one of them writing, with no synchronization between them (no "happens before" established by a lock, an atomic or a `join`). A data race is **undefined behaviour**, not merely "the result may be wrong". **This program has a bug:**

```cpp title="race.cpp" sanitize="thread" expect="fail"
#include <cstdio>
#include <thread>

int main() {
  long tokens = 0;
  auto work = [&] {
    for (int i = 0; i < 100000; ++i) ++tokens;   // two threads reading and writing one variable without synchronization
  };
  std::thread a(work), b(work);
  a.join();
  b.join();
  std::printf("%ld\n", tokens);
}
```

```text title="the ThreadSanitizer report (excerpt)"
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 8 at 0x7ffd... by thread T2:
    #0 main::{lambda()#1}::operator()() const race.cpp:7
  Previous write of size 8 at 0x7ffd... by thread T1:
    #0 main::{lambda()#1}::operator()() const race.cpp:7
```

Without TSan this program most likely prints something below 200000 (the two threads' `++` overwriting each other), and with `-O2` it may even print 200000, the compiler having optimized the loop into a single addition. That is UB: you cannot conclude the code is right from "the output looks right".

**A race condition** is the broader notion: the result depends on the timing of the threads. A program free of data races can still have a race condition (check-then-act, say: a free block was seen and taken by another thread before it could be allocated). TSan catches only data races, and race conditions are a matter of design: put the "check" and the "act" in the same critical section.

## Mutexes {#互斥锁}

Two ways to fix a data race: protect it with a lock, or use an atomic (the next chapter).

```cpp title="mutex_counter.cpp" sanitize="thread"
#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

int main() {
  long by_mutex = 0;
  std::mutex m;
  std::atomic<long> by_atomic{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < 10000; ++i) {
        {
          std::lock_guard lk(m);   // locks on construction and unlocks on leaving the scope (RAII)
          ++by_mutex;
        }
        by_atomic.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& t : ts) t.join();
  std::printf("mutex：%ld，atomic：%ld\n", by_mutex, by_atomic.load());
}
```

```text title="output"
mutex：40000，atomic：40000
```

The three lock guards:

| | Use |
| --- | --- |
| `std::lock_guard` | the simplest: hold one lock for the scope |
| `std::unique_lock` | can unlock partway, lock again and transfer ownership; **a condition variable needs it** |
| `std::scoped_lock` | locks several mutexes at once with a deadlock-free algorithm (C++17) |

A few principles:

- **the shorter the critical section the better**. Do not do I/O, allocate a large block, launch a kernel or call a user callback while holding a lock: the waiting turns straight into tail latency;
- **a lock protects data, not code**. For every shared variable you should be able to say "which lock protects it", preferably in a comment at its declaration, or by wrapping the data and the lock in one class;
- data read far more often than written (a model registry, a routing table) can use a `std::shared_mutex`: several readers hold it at once through `std::shared_lock` and a writer holds it exclusively through `std::unique_lock`.

## Deadlock {#死锁}

![Figure: the standard way to deadlock - two threads taking two locks in opposite orders](../assets/figures/deadlock-order.svg){.aig-svg}

Two threads taking two locks in opposite orders can end up each waiting for the lock the other holds, which is a deadlock. **This program has a bug** (this run happened not to deadlock, but TSan sees the hazard in the lock order):

```cpp title="lock_order.cpp" sanitize="thread" expect="fail"
#include <mutex>
#include <thread>

std::mutex cache_mu, sched_mu;

void evict() {        // the cache lock first, then the scheduler lock
  std::lock_guard a(cache_mu);
  std::lock_guard b(sched_mu);
}
void schedule() {     // the other order
  std::lock_guard a(sched_mu);
  std::lock_guard b(cache_mu);
}

int main() {
  std::thread t1(evict);
  t1.join();
  std::thread t2(schedule);
  t2.join();
}
```

```text title="the ThreadSanitizer report (excerpt)"
WARNING: ThreadSanitizer: lock-order-inversion (potential deadlock) (pid=12345)
  Cycle in lock order graph: M0 (cache_mu) => M1 (sched_mu) => M0
```

Ways to avoid deadlock:

- **a fixed lock order**: define a global hierarchy of locks (always the scheduler before the cache, say) and have all the code obey it;
- **take several locks with `std::scoped_lock(cache_mu, sched_mu)`**, which guarantees the order cannot deadlock;
- **call no external code while holding a lock** (a callback, a virtual function), since you do not know what it will lock;
- the most fundamental one: **share less**. Have each component touch only its own data in its own thread, and pass messages between components through queues.

## Condition variables and a blocking queue {#条件变量与阻塞队列}

"One thread produces and another consumes" is the commonest structure in an inference service: the network thread hands requests to the scheduler and the scheduler hands generated tokens to the detokenizer. A consumer with no data should **sleep** rather than spin, which is what condition variables are for.

Their use is fixed:

1. hold the mutex and check the condition;
2. if it does not hold, `cv.wait(lock, predicate)`: it atomically releases the lock and sleeps, and on waking retakes the lock and **checks the predicate again**, going back to sleep if it still does not hold;
3. whoever changes the condition calls `notify_one()` / `notify_all()` after modifying the data.

The predicate is required, because there are **spurious wakeups** (waking with nobody notifying), and because another consumer may have taken the element before this one woke.

Below is a blocking queue that can be closed. After `close()` a producer can push no more, and a consumer gets a `std::nullopt` once the remaining elements are drained, so it can leave its loop cleanly:

```cpp title="blocking_queue.hpp"
#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

template <class T>
class BlockingQueue {
 public:
  bool push(T v) {                 // returns false once closed
    {
      std::lock_guard lk(m_);
      if (closed_) return false;
      q_.push_back(std::move(v));
    }
    cv_.notify_one();              // notify outside the lock: the thread woken need not wait for it immediately
    return true;
  }
  std::optional<T> pop() {         // blocks while the queue is empty; returns nullopt once closed and drained
    std::unique_lock lk(m_);
    cv_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return std::nullopt;
    T v = std::move(q_.front());
    q_.pop_front();
    return v;
  }
  void close() {
    {
      std::lock_guard lk(m_);
      closed_ = true;
    }
    cv_.notify_all();              // wake every waiting consumer so they see "closed"
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<T> q_;                // protected by m_
  bool closed_ = false;            // protected by m_
};
```

One producer and two consumers:

```cpp title="producer_consumer.cpp" sanitize="thread"
#include <cstdio>
#include <thread>
#include <vector>

#include "blocking_queue.hpp"

struct Req {
  int id;
  int tokens;
};

int main() {
  BlockingQueue<Req> q;
  std::thread producer([&] {
    for (int i = 0; i < 100; ++i) q.push({i, i % 10 + 1});
    q.close();                     // production finished
  });
  std::vector<long> handled(2, 0), tokens(2, 0);   // each consumer writes only its own slot: nothing shared
  std::vector<std::thread> consumers;
  for (int c = 0; c < 2; ++c) {
    consumers.emplace_back([&, c] {
      while (auto r = q.pop()) {
        ++handled[c];
        tokens[c] += r->tokens;
      }
    });
  }
  producer.join();
  for (auto& t : consumers) t.join();
  std::printf("处理了 %ld 个请求，共 %ld 个 token\n", handled[0] + handled[1], tokens[0] + tokens[1]);
}
```

```text title="output"
处理了 100 个请求，共 550 个 token
```

`handled` and `tokens` are written by two threads at once, but each thread writes only the element at its own index: different memory locations, so no data race (they sit close together, so there is false sharing, and statistics updated often should be separated by a cache line, see [object layout, alignment and the cache](../memory/layout.md#伪共享)).
The main thread reads after the `join()`, and the `join()` establishes "the writes in the thread happen before join returns", so the read is safe.

## `std::jthread` and cooperative stopping {#stdjthread-与协作式停止}

A background thread (reporting metrics, a heartbeat, KV offloading) is usually a loop that needs a clean way to exit. C++20's `std::jthread` carries a **stop token**: the thread function's first parameter may be a `std::stop_token`, and once `request_stop()` is called from outside (or automatically when the `jthread` is destroyed), `stop_requested()` becomes true:

```cpp title="jthread_worker.cpp" sanitize="thread"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stop_token>
#include <thread>

int main() {
  std::atomic<int> ticks{0};
  {
    std::jthread reporter([&](std::stop_token st) {
      while (!st.stop_requested()) {
        ticks.fetch_add(1);        // pretending to report a metric
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      std::printf("reporter 收到停止请求，退出\n");
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }   // the jthread's destructor: request_stop() first, then join()
  std::printf("上报过：%s\n", ticks.load() > 0 ? "是" : "否");
}
```

```text title="output"
reporter 收到停止请求，退出
上报过：是
```

When the thread sleeps on a condition variable, use `std::condition_variable_any`'s `wait(lock, stop_token, predicate)` overload and a stop request wakes it.

## Threads or processes {#线程还是进程}

In a C++ service, threads of one process share memory and communicate most cheaply; a Python service is bounded by the GIL, so SGLang and vLLM both split the tokenizer, the scheduler and the detokenizer into **separate processes** passing messages over ZMQ (see mini-sglang's [messages and ZMQ](minisgl://serve/message/)).
What the two structures have in common is **replacing shared state with message passing**: each component owns its data and only exchanges messages through queues. That is far easier to get right than careful locking over shared data.

!!! interview "Answering in an interview"
    On multithreading: a data race (two threads touching the same memory without synchronization, at least one writing) is undefined behaviour; a race condition is a logical error where the result depends on the order, and can exist without a data race. Using locks: `lock_guard` is the simplest, `unique_lock` can unlock and pairs with a condition variable, and `scoped_lock` takes several locks at once without deadlocking; keep the critical section short and do no I/O or callbacks inside it; take several locks in a fixed order. A condition variable's `wait` has to carry a predicate (or sit in a `while`), because of spurious wakeups and missed notifications; `notify_all` when closing a queue. A background thread exits cooperatively through `std::jthread` and a `stop_token`. All concurrent code has to have been run under TSan.

## Exercises {#练习}

1. Give `BlockingQueue` a capacity limit: `push` blocks when the queue is full until there is room or the queue is closed (this is **backpressure**: when the downstream cannot keep up, slow the upstream down rather than piling requests up without limit). Write a test: capacity 4, a producer pushing 20 quickly, a consumer sleeping 1 ms per element, checking that the queue's length never exceeded 4 and that all 20 were handled. It has to be clean under TSan.

??? success "Answer"
    Two condition variables are needed: "not empty" for the consumers to wait on and "not full" for the producers. `close()` has to wake both:

    ```cpp title="bounded_queue.cpp" sanitize="thread"
    #include <algorithm>
    #include <chrono>
    #include <condition_variable>
    #include <cstdio>
    #include <deque>
    #include <mutex>
    #include <optional>
    #include <thread>

    template <class T>
    class BoundedQueue {
     public:
      explicit BoundedQueue(std::size_t cap) : cap_(cap) {}
      bool push(T v) {
        std::unique_lock lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return false;
        q_.push_back(std::move(v));
        max_seen_ = std::max(max_seen_, q_.size());
        lk.unlock();
        not_empty_.notify_one();
        return true;
      }
      std::optional<T> pop() {
        std::unique_lock lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return std::nullopt;
        T v = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        not_full_.notify_one();   // a slot has opened up
        return v;
      }
      void close() {
        {
          std::lock_guard lk(m_);
          closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
      }
      std::size_t max_seen() {
        std::lock_guard lk(m_);
        return max_seen_;
      }

     private:
      std::mutex m_;
      std::condition_variable not_empty_, not_full_;
      std::deque<T> q_;
      std::size_t cap_, max_seen_ = 0;
      bool closed_ = false;
    };

    int main() {
      BoundedQueue<int> q(4);
      int handled = 0;
      std::thread consumer([&] {
        while (auto v = q.pop()) {
          ++handled;
          std::this_thread::sleep_for(std::chrono::milliseconds(1));   // a slow consumer
        }
      });
      for (int i = 0; i < 20; ++i) q.push(i);
      q.close();
      consumer.join();
      std::printf("处理 %d 个，队列长度从未超过 4：%s\n", handled, q.max_seen() <= 4 ? "是" : "否");
    }
    ```

    ```text title="output"
    处理 20 个，队列长度从未超过 4：是
    ```

2. The consumer code below has two problems. Name them:

    ```cpp
    std::unique_lock lk(m);
    if (q.empty()) cv.wait(lk);
    auto item = q.front();
    q.pop_front();
    ```

??? success "Answer"
    1. The `if` checks only once: after a spurious wakeup, or when another consumer took the element before this one woke, `q.front()` is called on an empty queue (undefined behaviour). It needs `cv.wait(lk, [&] { return !q.empty(); })` or `while (q.empty()) cv.wait(lk);`;
    2. There is no handling of "closed": once the producer finishes, the consumer sleeps in `wait` forever and the thread cannot exit. The predicate has to include `closed`, and it has to return when the queue is empty and closed.

## Summary {#小结}

- [x] A data race is undefined behaviour; check all concurrent code with TSan (`-fsanitize=thread`), since "the output looks right" proves nothing.
- [x] A lock protects data: keep the critical section short and do no I/O or callbacks inside it; take several locks with `scoped_lock` or in a fixed order to avoid deadlock.
- [x] A condition variable pairs with `unique_lock` and its `wait` has to carry a predicate; `notify_all` when closing a queue so every waiter leaves.
- [x] A background thread exits cooperatively through `std::jthread` and a `stop_token`.
- [x] The best concurrent design shares less: each component owns its data and passes messages through queues.
