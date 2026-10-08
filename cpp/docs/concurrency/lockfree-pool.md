# 无锁队列与线程池

<p class="lead">有了原子操作和内存序，就可以写出推理系统里最常用的两个并发组件：单生产者单消费者的无锁环形队列（流水线各阶段之间传数据，没有锁、也没有系统调用），以及线程池（把 CPU 上的并行工作——分词、加载权重分片、组织 batch——分给固定的几个线程）。这一章把它们各写一遍，并在 ThreadSanitizer 下验证。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 单生产者单消费者（SPSC）队列为什么可以不用锁？它的读写指针分别由谁来写？
    2. 环形队列的容量为什么通常取 2 的幂？
    3. 线程池的 `submit` 怎样把任务的返回值和异常交还给调用者？
    4. 线程池析构时，队列里还没执行的任务怎么办？
    5. 在线程池的任务里等待另一个提交到**同一个池**的任务，可能出什么问题？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 只有一个线程写每一个指针：生产者只写 `tail`（写到哪了），消费者只写 `head`（读到哪了），不存在两个线程同时修改同一个变量。生产者写好元素再用 release 更新 `tail`，消费者用 acquire 读 `tail` 再读元素，就能安全地交接数据；反方向同理。
    2. 下标对容量取模可以写成按位与（`index & (Cap - 1)`），比除法快；指针一直递增、只在访问数组时取模，也能区分"满"和"空"。
    3. 用 `std::packaged_task` 包装任务、返回它的 `std::future`：工作线程执行任务时，返回值或抛出的异常都被存进共享状态，调用方 `future.get()` 时拿到结果或者重新抛出异常。
    4. 要明确策略：通常先设置停止标志、唤醒所有线程，让它们把队列里剩下的任务执行完再退出（或者直接丢弃），最后 `join` 所有线程。不能让线程在析构之后还访问已经销毁的队列。
    5. 线程池饥饿死锁：如果所有工作线程都在等待还在队列里的任务，就没有线程去执行它们，永远等下去。不要在池里的任务中阻塞等待同一个池的任务，或者用单独的池、改成异步的延续。

## 单生产者单消费者环形队列

![图：环形缓冲区——生产者写 head、消费者读 tail](../assets/figures/ring-buffer.svg){.aig-svg}

推理服务的流水线里，很多队列天然只有一个写者和一个读者：调度器线程把采样出的 token 交给 detokenizer 线程，网络线程把请求交给 tokenizer 线程，GPU 完成事件的轮询线程把完成通知交给调度器。这种队列可以做到**完全无锁**：

- 一个固定大小的数组，一个"写到哪了"的 `tail`（只有生产者写），一个"读到哪了"的 `head`（只有消费者写）；
- 生产者写入元素后，用 **release** 更新 `tail`；消费者用 **acquire** 读 `tail`，就能安全地读到元素——这正是上一章的"发布"模式；反方向同理，消费者用 release 更新 `head`，告诉生产者这个槽位可以复用了；
- `head` 和 `tail` 只增不减，用 `index & (容量 - 1)` 映射到数组下标（所以容量取 2 的幂，取模变成一次按位与），`tail - head` 就是当前的元素个数。

再加两个优化：`head` 和 `tail` 放在不同的缓存行（否则生产者和消费者会伪共享）；各自**缓存**一份对方的指针，只有在"看起来满了 / 空了"时才去读对方的原子变量，减少跨核的缓存行传递。

```cpp title="spsc_queue.hpp"
#pragma once
#include <array>
#include <atomic>
#include <cstddef>

template <class T, std::size_t Cap>
class SpscQueue {
  static_assert(Cap > 0 && (Cap & (Cap - 1)) == 0, "容量必须是 2 的幂");

 public:
  bool try_push(const T& v) {   // 只能由生产者线程调用
    const std::size_t t = prod_.tail.load(std::memory_order_relaxed);
    if (t - prod_.head_cache == Cap) {                                 // 看起来满了：刷新消费者的进度
      prod_.head_cache = cons_.head.load(std::memory_order_acquire);
      if (t - prod_.head_cache == Cap) return false;
    }
    buf_[t & (Cap - 1)] = v;
    prod_.tail.store(t + 1, std::memory_order_release);                // 先写数据，再发布
    return true;
  }
  bool try_pop(T& out) {        // 只能由消费者线程调用
    const std::size_t h = cons_.head.load(std::memory_order_relaxed);
    if (h == cons_.tail_cache) {                                       // 看起来空了：刷新生产者的进度
      cons_.tail_cache = prod_.tail.load(std::memory_order_acquire);
      if (h == cons_.tail_cache) return false;
    }
    out = buf_[h & (Cap - 1)];
    cons_.head.store(h + 1, std::memory_order_release);                // 读完了，这个槽位可以复用
    return true;
  }

 private:
  struct alignas(64) Producer {       // 生产者写的东西放在一个缓存行
    std::atomic<std::size_t> tail{0};
    std::size_t head_cache = 0;
  };
  struct alignas(64) Consumer {       // 消费者写的东西放在另一个缓存行
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
      while (!q.try_push(i)) std::this_thread::yield();   // 满了就让出 CPU
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

```text title="输出"
收到 200000 个，顺序正确，和 = 20000100000
```

这个队列没有锁、没有系统调用，每次操作通常只有一次原子读写，一次操作可以做到几纳秒。代价是**空的时候消费者要自己决定怎么等**：一直自旋（延迟最低，但占满一个核）、`yield`、或者自旋一会儿后退回到条件变量 / 原子等待。延迟敏感的场景（比如 RDMA 完成队列的轮询线程）通常绑一个核专门自旋。

!!! note "多生产者或多消费者呢？"
    多生产者多消费者的无锁队列（比如 Dmitry Vyukov 的有界 MPMC 队列）要用 CAS 抢槽位，正确性论证复杂得多。
    实践中的顺序是：先用上一章的互斥锁 + 条件变量队列；profile 证明锁是瓶颈后，看能不能把结构改成多个 SPSC 队列（每个生产者一个）；最后才考虑成熟库里的 MPMC 实现，而不是自己写。

## 线程池

![图：线程池——任务队列、工作线程、条件变量与停止标志](../assets/figures/thread-pool.svg){.aig-svg}

线程创建和销毁都有开销（几十微秒），而且线程数远超核数时，上下文切换会吃掉大量时间。线程池预先创建固定数量的工作线程，任务提交到一个共享队列，由空闲的线程取走执行。

实现要点：

- 任务队列用上一章的"互斥锁 + 条件变量"；
- `submit` 返回一个 `std::future`：用 `std::packaged_task` 包装任务，它把返回值**和异常**都存进共享状态，调用者 `get()` 时拿到结果，或者异常在调用者的线程里重新抛出；
- **在锁外执行任务**，否则任务之间就串行了；
- 析构时先设置"停止"标志并唤醒所有线程，工作线程把队列里剩下的任务做完再退出（也可以选择丢弃），最后 `join`。

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
  }   // 接着 workers_ 析构（jthread 自动 join）：工作线程做完剩下的任务后退出

  template <class F>
  auto submit(F f) -> std::future<std::invoke_result_t<F>> {
    using R = std::invoke_result_t<F>;
    // packaged_task 只能移动，而 std::function 要求可拷贝，所以放进 shared_ptr
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
        if (tasks_.empty()) return;   // 已经停止，且队列取空了
        job = std::move(tasks_.front());
        tasks_.pop_front();
      }
      job();                          // 在锁外执行
    }
  }

  std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;   // 由 m_ 保护
  bool stopping_ = false;                     // 由 m_ 保护
  std::vector<std::jthread> workers_;         // 最后声明、最先析构：join 时上面的成员都还活着
};
```

成员的声明顺序在这里很重要：`workers_` 必须最后声明，这样它**最先**析构，工作线程在 `join` 之前还能安全地使用锁、条件变量和队列（回忆[值语义与 RAII](../basics/value-raii.md) 里"成员按声明的相反顺序析构"）。

```cpp title="pool_demo.cpp" sanitize="thread"
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "thread_pool.hpp"

// 把 [0, n) 切成若干段交给线程池，等所有段完成
template <class F>
void parallel_for(ThreadPool& pool, int n, int chunks, F f) {
  std::vector<std::future<void>> fs;
  for (int c = 0; c < chunks; ++c) {
    int lo = n * c / chunks, hi = n * (c + 1) / chunks;
    fs.push_back(pool.submit([=] {
      for (int i = lo; i < hi; ++i) f(i);
    }));
  }
  for (auto& fu : fs) fu.get();   // 任何一段抛出的异常会在这里重新抛出
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

  std::vector<int> shard_tokens(8);   // 每个分片由一个任务写自己的那一格
  parallel_for(pool, 8, 3, [&](int i) { shard_tokens[i] = (i + 1) * 1000; });
  std::printf("8 个分片共 %d 个 token\n", std::accumulate(shard_tokens.begin(), shard_tokens.end(), 0));
}
```

```text title="输出"
平方和 = 328350
任务里的异常在 get() 时抛出：分词失败：非法的 UTF-8
8 个分片共 36000 个 token
```

`future::get()` 除了取结果，还建立了同步：任务里对 `shard_tokens` 的写入，在 `get()` 返回后对主线程可见，所以主线程读它是安全的，TSan 也不会报告。

### 线程池的几个坑

- **在任务里等待同一个池的另一个任务**：如果所有工作线程都在等"还在队列里、没人执行"的任务，就永远等下去了（线程池饥饿死锁）。任务之间有依赖时，要么用不同的池，要么让等待方自己去执行队列里的任务（工作窃取的做法）；
- **阻塞 I/O 放进计算池**：一个读网络的任务会占住一个线程，计算任务跟着排队。I/O 用单独的池或者异步 I/O；
- **超额订阅**：线程池开了 32 个线程，每个任务里又调用了 PyTorch 或 OpenMP，它们各自再开 32 个线程，1024 个线程抢 32 个核，吞吐反而大跌。常见的修法是限制内层的线程数（`torch.set_num_threads`、`OMP_NUM_THREADS`）；
- **任务太小**：每个任务要一次加锁、一次唤醒，任务本身只有几微秒时，调度开销就占了大头。把工作按块切分（上面的 `parallel_for` 把 8 个元素切成 3 段），或者用每个线程一个本地队列、空闲时去偷别人任务的**工作窃取**结构（TBB、Rust 的 rayon、Go 的调度器都是这样）。

!!! interview "怎么讲清楚"
    讲无锁队列与线程池：单生产者单消费者的环形队列可以完全无锁——生产者只写 `tail`、消费者只写 `head`，写完数据再 release 移动 `tail`，读完再 release 移动 `head`，对方用 acquire 读；容量取 2 的幂，下标用按位与；两个指针放在不同的缓存行，各自缓存对方的值减少争用。多生产者多消费者先用锁队列，确认是瓶颈再优化。线程池：锁 + 条件变量的任务队列，`packaged_task` 把返回值和异常交给 `future`，任务在锁外执行，析构时先置停止标志、唤醒所有线程再 join。任务里等待同一个池里的另一个任务会饥饿死锁。

## 练习

1. 给 `ThreadPool` 加一个 `wait_idle()`：阻塞直到队列为空、且没有正在执行的任务。用它改写一个"提交 1000 个任务、不保存 future、最后等全部完成"的程序，要求在 TSan 下没有报告。

??? success "参考答案"
    维护一个"未完成任务数"：提交时加一，任务执行完减一，减到 0 时通知等待者。计数和队列用同一把锁保护，并用第二个条件变量通知 `wait_idle`：

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
      int pending_ = 0;   // 已提交、还没执行完的任务数
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

    ```text title="输出"
    wait_idle 返回时完成了 1000 个任务
    ```

    注意 `pending_` 是在**任务执行完之后**才减的，而不是取出任务时：否则 `wait_idle` 可能在最后几个任务还在执行时就返回。

2. `SpscQueue` 里，如果把 `try_push` 中 `tail` 的 `store` 改成 `relaxed`，会发生什么？如果把 `Producer` 和 `Consumer` 的 `alignas(64)` 去掉呢？

??? success "参考答案"
    `tail` 改成 relaxed 之后，写入元素和更新 `tail` 之间失去了顺序保证：消费者可能先看到新的 `tail`，读到一个还没写好的槽位——数据竞争，TSan 会报告，ARM 上会真的读到旧数据。
    去掉 `alignas(64)` 不影响正确性，但生产者频繁写的 `tail` 和消费者频繁写的 `head` 很可能落在同一个缓存行里，每次操作都让对方的缓存行失效，吞吐会下降数倍（见[伪共享](../memory/layout.md#伪共享)）。

## 小结

- [x] 单生产者单消费者队列可以完全无锁：生产者只写 `tail`、消费者只写 `head`，用 release / acquire 发布数据和空出槽位。
- [x] 容量取 2 的幂，下标用按位与；两端的指针放在不同的缓存行，各自缓存对方的指针。
- [x] 多生产者多消费者先用锁队列，确认是瓶颈再考虑拆成多个 SPSC 或用成熟的实现。
- [x] 线程池：锁 + 条件变量的任务队列，`packaged_task` 把返回值和异常交还给 `future`，任务在锁外执行，析构时先停止再 join。
- [x] 小心线程池饥饿死锁、阻塞 I/O、超额订阅和过小的任务。
