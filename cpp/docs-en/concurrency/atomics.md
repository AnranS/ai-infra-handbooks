# Atomics and memory order

<p class="lead">Locks solve most concurrency problems, but the hottest paths, statistics counters, a "the data is ready" flag, a lock-free queue's read and write pointers, signals between a CPU and a GPU or between two cards, call for atomics. The atomic operations themselves are not hard; what is hard is <strong>memory order</strong>: when and in what order does another thread see what one thread wrote. This chapter covers relaxed, acquire / release and seq_cst with as few concepts as possible, and maps them onto the signalling in GPU communication libraries.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. One thread writes the data and then sets `ready` to `true`; another reads the data once it sees `ready == true`. Is `std::atomic<bool>` on `ready` enough? Which memory order?
    2. Which memory order does a request counter take? Why?
    3. Why is `compare_exchange_weak` usually written in a loop? What does it do on failure?
    4. Why is a spinlock's `lock()` acquire and its `unlock()` release?
    5. On a GPU, a kernel finishes writing a block of data and sets a flag to notify another card. Why is a fence needed in between?

??? success "Answers (try it yourself first, then expand)"
    1. Atomicity is enough but ordering is not: it also has to be guaranteed that "seeing `ready == true` means seeing the data already written". The writer uses `ready.store(true, release)` and the reader `ready.load(acquire)` (or both use the default `seq_cst`).
    2. `relaxed`: the counter needs only atomicity and does not synchronize any other data; the final read happens after the `join()`, which already provides the synchronization.
    3. It is allowed to "fail spuriously" (returning failure even when the value matches), and another thread may have changed the value concurrently, so it retries in a loop. On failure it updates `expected` to the actual current value, which the next round uses to compute the new one.
    4. `lock()` is acquire: having taken the lock, it has to see everything the previous holder wrote in the critical section; `unlock()` is release: the writes in the critical section have to be visible to the next holder before the lock is released.
    5. A GPU's writes may be reordered or still sit in a cache: without a fence, the other side may see the flag before the data. `__threadfence_system()` (or a write with release semantics) guarantees the data is visible to other devices before the flag is written, which is the same thing as release / acquire on a CPU.

## Atomic operations {#原子操作}

Every operation on a `std::atomic<T>` is indivisible: no "read halfway through while someone writes", and several threads `fetch_add`ing it at once lose no updates. The common operations:

| Operation | Meaning |
| --- | --- |
| `load()` / `store(v)` | read / write |
| `fetch_add(d)` / `fetch_sub(d)` | add or subtract, returning the old value |
| `exchange(v)` | write the new value, returning the old one |
| `compare_exchange_weak/strong(expected, desired)` | if the current value equals `expected`, set it to `desired` and return `true`; otherwise write the current value into `expected` and return `false` (CAS) |
| `wait(old)` / `notify_one()` / `notify_all()` | C++20: block until the value is no longer `old` (without a condition variable) |

Atomic types the size of an `int`, a `long` or a pointer are **lock-free** on the mainstream platforms (`std::atomic<T>::is_always_lock_free`), compiling into a single `lock`-prefixed instruction or an ordinary load or store.

## Why memory order is needed as well {#为什么还需要内存序}

![Figure: the difference between relaxed and release/acquire](../assets/figures/memory-order.svg){.aig-svg}

Atomicity only guarantees that the operations on **that one variable** are not torn. But both the compiler and the CPU reorder instructions: as long as a single thread's result is unchanged, writing the data and writing the flag may be swapped, and another thread may then see the flag before the data.

In the "publish a configuration" example below, `ready` is atomic but uses `relaxed` (atomicity only, no ordering). **This program has a bug:**

```cpp title="publish_relaxed.cpp" sanitize="thread" expect="fail"
#include <atomic>
#include <cstdio>
#include <thread>

struct Config {
  int max_batch;
  int page_size;
};
Config config;                    // an ordinary variable
std::atomic<bool> ready{false};

int main() {
  std::thread loader([] {
    config = {256, 16};
    ready.store(true, std::memory_order_relaxed);   // atomic only, with no guarantee that config was written first
  });
  std::thread worker([] {
    while (!ready.load(std::memory_order_relaxed)) {
    }
    std::printf("%d\n", config.max_batch);           // may read a stale value: a data race
  });
  loader.join();
  worker.join();
}
```

```text title="the ThreadSanitizer report (excerpt)"
WARNING: ThreadSanitizer: data race (pid=12345)
  Read of size 4 at 0x55... by thread T2:
    #0 main::{lambda()#2}::operator()() const publish_relaxed.cpp:20
  Previous write of size 8 at 0x55... by thread T1:
    #0 main::{lambda()#1}::operator()() const publish_relaxed.cpp:14
```

On x86 this program "almost always" works (x86's hardware memory model is relatively strong), while on an ARM server (Grace, Kunpeng, Graviton) it really can read a stale value, and the compiler is entitled to reorder on any platform.

## acquire / release: publishing and acquiring {#acquire--release发布与获取}

The fix is to pair a **release write** with an **acquire read**:

- `store(v, memory_order_release)`: no read or write **before** it may be reordered after it;
- `load(memory_order_acquire)`: no read or write **after** it may be reordered before it;
- when an acquire read **sees** the value a release write stored, everything the writing thread did before the release is visible to the reading thread after the acquire. The release write is said to "synchronize with" the acquire read.

```cpp title="publish_acquire.cpp" sanitize="thread"
#include <atomic>
#include <cstdio>
#include <thread>

struct Config {
  int max_batch;
  int page_size;
};
Config config;
std::atomic<bool> ready{false};

int main() {
  std::thread loader([] {
    config = {256, 16};
    ready.store(true, std::memory_order_release);    // everything written before (config) is "published"
  });
  std::thread worker([] {
    while (!ready.load(std::memory_order_acquire)) { // once true is read, config is certainly the new one
    }
    std::printf("max_batch=%d page_size=%d\n", config.max_batch, config.page_size);
  });
  loader.join();
  worker.join();
}
```

```text title="output"
max_batch=256 page_size=16
```

This is the most important pattern of all: **write the data first, then set the flag with a release; read the flag with an acquire, then read the data**. Lock-free queues, a "the weights are loaded" notification and handing a pointer between threads are all this.

## Which memory order to use {#该用哪种内存序}

| Memory order | Guarantee | Typical use |
| --- | --- | --- |
| `relaxed` | only that the operations on this variable are atomic | statistics counters (request counts, token counts), accumulation where only the final value matters, an ID generator |
| `acquire` / `release` | used in pairs, establishing "happens before" | publishing data (a flag plus the data), locks, a lock-free queue's read and write pointers |
| `seq_cst` (the default) | all seq_cst operations have one globally consistent order | when in doubt; algorithms that need a global order among several flags |

A few suggestions:

- the default without a memory order is `seq_cst`, which is always correct and merely possibly slower (a seq_cst store on x86 needs an `xchg` or an `mfence`). **Get it right first and relax it once it is proved to be the bottleneck**;
- `relaxed` on a counter is safe, provided nobody infers "some other data is ready" from the counter having changed;
- every use of a non-default memory order deserves a line of comment saying what it pairs with.

## A CAS loop {#cas-循环}

`compare_exchange` is the basic tool for building lock-free algorithms: read the current value, compute the new one, and write it **only if nobody changed it in between**, retrying with the latest value otherwise. Updating a "peak memory use" concurrently, say:

```cpp title="atomic_max.cpp" sanitize="thread"
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

void atomic_max(std::atomic<long>& target, long v) {
  long cur = target.load(std::memory_order_relaxed);
  while (cur < v && !target.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    // on failure cur already holds the current value, so the loop compares again
  }
}

int main() {
  std::atomic<long> peak{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t) {
    ts.emplace_back([&, t] {
      for (long i = 0; i < 10000; ++i) atomic_max(peak, (i * 7 + t * 13) % 9973);
    });
  }
  for (auto& th : ts) th.join();
  std::printf("峰值 = %ld\n", peak.load());
}
```

```text title="output"
峰值 = 9972
```

The `weak` version allows a "spurious failure" (returning `false` even when the values are equal, which is more efficient on platforms like ARM that implement CAS with LL/SC), so it is always written in a loop; outside a loop, use `strong`.

!!! warning "The ABA problem"
    CAS only compares values. If a thread reads `A` while another changes it to `B` and back to `A`, the CAS still succeeds while the change in between is missed.
    In a lock-free stack of pointers that means "a node at the same address was freed and allocated again", the classic ABA problem. The fixes are CASing a version number together with the pointer, or hazard pointers or epoch-based reclamation. In practice, where a single-producer single-consumer ring buffer will do (the next chapter), do not write a multi-producer lock-free linked structure yourself.

## A spinlock {#自旋锁}

One atomic flag is enough for the simplest lock: `lock()` changes the flag from `false` to `true` with acquire and `unlock()` changes it back with release. acquire / release is exactly what keeps the critical section's reads and writes from "leaking" outside the lock:

```cpp title="spinlock.cpp" sanitize="thread"
#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

class SpinLock {
 public:
  void lock() {
    while (flag_.test_and_set(std::memory_order_acquire)) {   // take the lock
      while (flag_.test(std::memory_order_relaxed)) {         // while it cannot be taken, only read and never write, so the cache line does not bounce
      }
    }
  }
  void unlock() { flag_.clear(std::memory_order_release); }

 private:
  std::atomic_flag flag_;   // from C++20 it is default-initialized to "clear"
};

int main() {
  SpinLock lock;
  long counter = 0;
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < 10000; ++i) {
        std::lock_guard g(lock);   // any type with lock()/unlock() works with lock_guard
        ++counter;
      }
    });
  }
  for (auto& th : ts) th.join();
  std::printf("counter = %ld\n", counter);
}
```

```text title="output"
counter = 40000
```

A spinlock only suits **a very short critical section with no more threads than cores**: a thread that cannot take the lock burns the CPU spinning, and when the holder is descheduled by the operating system the others spin away a whole time slice for nothing. Ordinarily use a `std::mutex` (modern implementations spin briefly before sleeping).

## C++20's atomic wait {#c20-的原子等待}

To "wait for a flag to change" without spinning, C++20's `wait` / `notify` sleep on the atomic variable itself (a futex on Linux), which is lighter than a condition variable:

```cpp title="atomic_wait.cpp" sanitize="thread"
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
  std::atomic<bool> weights_loaded{false};
  std::atomic<int> started{0};
  std::vector<std::thread> workers;
  for (int i = 0; i < 3; ++i) {
    workers.emplace_back([&] {
      weights_loaded.wait(false, std::memory_order_acquire);   // sleep while the value is still false
      started.fetch_add(1, std::memory_order_relaxed);
    });
  }
  weights_loaded.store(true, std::memory_order_release);        // loading finished
  weights_loaded.notify_all();
  for (auto& w : workers) w.join();
  std::printf("%d 个工作线程都开始了\n", started.load());
}
```

```text title="output"
3 个工作线程都开始了
```

## The GPU counterpart {#和-gpu-的对应}

The same problem exists on a GPU in exactly the same form, only at a larger scale:

- one block writes shared memory and another thread reads it, which needs `__syncthreads()`;
- a kernel writes global memory and then sets a flag with an atomic to notify **another block**, which needs a `__threadfence()` between the data and the flag (or an atomic with release semantics), with an acquire on the reading side; this is exactly what the CUDA handbook's [single-pass reduction](cuda://kernels/reduction/) has to handle when "the last block to finish combines the results";
- one card notifying another (NVSHMEM's `put` + `signal`, DeepEP's low-latency mode): write the data to the peer first, then write a signal; the peer polls the signal and only reads the data once it sees it. The fence in between is release / acquire's cross-device version.

Reading code like this, finding "where the data is written, where the flag is written, and what guarantees the order between them" makes the logic clear.

!!! interview "Answering in an interview"
    The core pattern of the memory order question: the producer writes the data and then writes the flag with a release; the consumer reads the flag with an acquire and then reads the data, and is guaranteed to see it. Merely making the flag `atomic` and using `relaxed` is not enough. A counter needs only atomicity and no ordering, so `relaxed`; when in doubt use the default `seq_cst` and verify with TSan. A CAS goes in a loop: `compare_exchange_weak` may fail spuriously and writes the current value into `expected` on failure; a multi-producer lock-free structure has to watch out for ABA. A spinlock's `lock` is acquire and its `unlock` is release, which keeps the critical section's reads and writes from moving outside the lock. A GPU's `__threadfence` and a cross-card put + signal are the same rules applied at a larger scale.

## Exercises {#练习}

1. Choose a memory order for each of these and say why:
    1. every thread accumulates "the total tokens processed", printed only at the end;
    2. a background thread finishes loading a LoRA adapter and stores a pointer to it in a `std::atomic<Adapter*>`, and a request thread uses it as soon as it reads a non-null pointer;
    3. the main thread sets `std::atomic<bool> stop` and a worker checks it in its loop, exiting when it sees `true` (it reads no other data the main thread wrote before exiting).

??? success "Answer"
    1. `relaxed`: atomicity is all that is needed, and the final read happens after the `join()`, which already provides the synchronization.
    2. `release` on the store and `acquire` on the load: the request thread has to see the adapter object's contents after it was fully constructed, the classic "publish". With `relaxed` alone it may see a state where "the pointer is visible but the object's contents are not written yet".
    3. `relaxed` is enough: the flag carries no other data, and the thread sees it at its next check at the latest. If it had to read data the main thread wrote before setting `stop` (the reason for stopping, say), it would need release / acquire.

2. Implement a single-use "latch" with a `std::atomic<int>`: after `count_down()` has been called N times, every thread in `wait()` is released. (C++20 already has `std::latch`; implement it here for yourself.) It has to be clean under TSan.

??? success "Answer"
    ```cpp title="my_latch.cpp" sanitize="thread"
    #include <atomic>
    #include <cstdio>
    #include <thread>
    #include <vector>

    class Latch {
     public:
      explicit Latch(int n) : count_(n) {}
      void count_down() {
        if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) count_.notify_all();   // the last one to arrive does the waking
      }
      void wait() {
        int c = count_.load(std::memory_order_acquire);
        while (c != 0) {
          count_.wait(c, std::memory_order_acquire);   // sleep while the value is still c
          c = count_.load(std::memory_order_acquire);
        }
      }

     private:
      std::atomic<int> count_;
    };

    int main() {
      const int n = 4;
      Latch ready(n);
      int loaded[n] = {};
      std::vector<std::thread> shards;
      for (int i = 0; i < n; ++i) {
        shards.emplace_back([&, i] {
          loaded[i] = 100 + i;   // each loads one shard of the weights
          ready.count_down();
        });
      }
      ready.wait();               // serving starts only once every shard is loaded
      int sum = 0;
      for (int v : loaded) sum += v;
      std::printf("所有分片就绪，校验和 %d\n", sum);
      for (auto& t : shards) t.join();
    }
    ```

    ```text title="output"
    所有分片就绪，校验和 406
    ```

    `fetch_sub` uses `acq_rel`: the release publishes the `loaded[i]` each thread wrote before its `count_down`, and the acquire lets the last thread see every earlier thread's writes; once `wait()` reads 0 with an acquire, the main thread can read `loaded` safely.

## Summary {#小结}

- [x] Atomicity only guarantees that a single variable is not torn; the visible order across variables comes from the memory order.
- [x] "Write the data, then write the flag with a release; read the flag with an acquire, then read the data" is the core pattern, and locks, lock-free queues and handing a pointer between threads all rest on it.
- [x] Counters take `relaxed`, publishing data takes acquire / release, and when in doubt take the default `seq_cst` and verify with TSan.
- [x] A CAS goes in a loop and `expected` is updated to the current value on failure; writing a multi-producer lock-free structure yourself means watching out for ABA.
- [x] A GPU's `__threadfence` and a cross-card put + signal are the same rules applied at a larger scale.
