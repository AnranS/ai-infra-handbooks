# 线程、锁与条件变量

<p class="lead">推理服务里到处是并发：tokenizer、调度器、detokenizer 各跑一个线程或进程，网络线程收请求，后台线程加载权重、上报指标，通信库里还有专门的代理线程。C++ 的线程是真正并行的，没有 GIL 兜底，任何两个线程无同步地读写同一个变量都是未定义行为。这一章讲线程、互斥锁、条件变量的正确用法，并用 ThreadSanitizer 证明代码没有数据竞争。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 什么是数据竞争？它和"竞态条件"有什么区别？
    2. `std::lock_guard`、`std::unique_lock`、`std::scoped_lock` 各用在什么地方？
    3. 条件变量的 `wait` 为什么一定要带一个谓词（或者写在 `while` 循环里）？
    4. 两个线程分别以不同的顺序获取两把锁，会发生什么？怎样避免？
    5. `std::jthread` 比 `std::thread` 多了什么？

## 线程与数据竞争

`std::thread` 创建一个线程，`join()` 等它结束。线程对象析构时如果还没有 `join()`（或 `detach()`），程序直接 `std::terminate`——所以 C++20 起优先用 `std::jthread`，它析构时会自动请求停止并 `join()`（见本章最后一节）。

**数据竞争**：两个线程访问同一个内存位置，至少一个是写，并且它们之间没有同步（锁、原子操作、`join` 等建立的"先于"关系）。数据竞争是**未定义行为**，不只是"结果可能不对"。**这个程序有 bug：**

```cpp title="race.cpp" sanitize="thread" expect="fail"
#include <cstdio>
#include <thread>

int main() {
  long tokens = 0;
  auto work = [&] {
    for (int i = 0; i < 100000; ++i) ++tokens;   // 两个线程无同步地读写同一个变量
  };
  std::thread a(work), b(work);
  a.join();
  b.join();
  std::printf("%ld\n", tokens);
}
```

```text title="ThreadSanitizer 的报告（节选）"
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 8 at 0x7ffd... by thread T2:
    #0 main::{lambda()#1}::operator()() const race.cpp:7
  Previous write of size 8 at 0x7ffd... by thread T1:
    #0 main::{lambda()#1}::operator()() const race.cpp:7
```

不开 TSan 时，这个程序大概率打印一个小于 200000 的数（两个线程的 `++` 互相覆盖），开了 `-O2` 甚至可能打印出 200000——编译器把循环优化成了一次加法。这就是 UB：你没法从"输出看起来对"推断出代码是对的。

**竞态条件**是更宽泛的概念：结果依赖于线程执行的时序。没有数据竞争的程序也可以有竞态条件（比如"先检查、再行动"：检查到还有空闲块，还没分配就被别的线程抢走了）。TSan 只能抓数据竞争，竞态条件要靠设计：把"检查"和"行动"放在同一个临界区里。

## 互斥锁

修复数据竞争的两种办法：用锁保护，或者用原子变量（下一章）。

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
          std::lock_guard lk(m);   // 构造时加锁，离开作用域时解锁（RAII）
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

```text title="输出"
mutex：40000，atomic：40000
```

三种锁守卫：

| | 用途 |
| --- | --- |
| `std::lock_guard` | 最简单：作用域内持有一把锁 |
| `std::unique_lock` | 可以中途解锁、再加锁、转移所有权；**条件变量必须配它** |
| `std::scoped_lock` | 同时锁住多把锁，内部用避免死锁的算法（C++17） |

几条原则：

- **临界区越短越好**。不要在持锁时做 I/O、分配大块内存、启动 kernel、调用用户回调——锁的等待时间会直接变成尾延迟；
- **锁保护的是数据，不是代码**。每个共享变量都应该能说清楚"它由哪把锁保护"，最好在声明处写注释，或者把数据和锁封装在一个类里；
- 读多写少的数据（模型注册表、路由表）可以用 `std::shared_mutex`：多个读者用 `std::shared_lock` 同时持有，写者用 `std::unique_lock` 独占。

## 死锁

两个线程以相反的顺序获取两把锁，就可能互相等待对方持有的锁——死锁。**这个程序有 bug**（这次运行碰巧没死锁，但 TSan 能从加锁顺序上看出隐患）：

```cpp title="lock_order.cpp" sanitize="thread" expect="fail"
#include <mutex>
#include <thread>

std::mutex cache_mu, sched_mu;

void evict() {        // 先拿缓存锁，再拿调度器锁
  std::lock_guard a(cache_mu);
  std::lock_guard b(sched_mu);
}
void schedule() {     // 顺序反过来
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

```text title="ThreadSanitizer 的报告（节选）"
WARNING: ThreadSanitizer: lock-order-inversion (potential deadlock) (pid=12345)
  Cycle in lock order graph: M0 (cache_mu) => M1 (sched_mu) => M0
```

避免死锁的办法：

- **固定的加锁顺序**：全局规定锁的层级（比如总是先调度器、后缓存），所有代码都遵守；
- **一次拿多把锁时用 `std::scoped_lock(cache_mu, sched_mu)`**，它保证不会因为顺序而死锁；
- **持有锁时不调用外部代码**（回调、虚函数），你不知道它会去拿什么锁；
- 最根本的：**减少共享**。让每个组件只在自己的线程里访问自己的数据，组件之间通过队列传递消息。

## 条件变量与阻塞队列

"一个线程产出、另一个线程消费"是推理服务里最常见的结构：网络线程把请求交给调度器，调度器把生成的 token 交给 detokenizer。消费者在没有数据时应该**睡眠**而不是空转，这就要用条件变量。

条件变量的用法固定：

1. 持有互斥锁，检查条件；
2. 条件不满足就 `cv.wait(lock, 谓词)`：它原子地释放锁并睡眠，被唤醒后重新拿到锁、**再检查一次谓词**，不满足就继续睡；
3. 改变条件的一方在修改数据后调用 `notify_one()` / `notify_all()`。

必须带谓词，因为存在**虚假唤醒**（没人通知也可能醒来），也可能醒来时条件已经被别的消费者抢先消费掉了。

下面是一个可以关闭的阻塞队列。`close()` 之后生产者不能再放，消费者取完剩下的元素后得到 `std::nullopt`，于是可以干净地退出循环：

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
  bool push(T v) {                 // 已关闭时返回 false
    {
      std::lock_guard lk(m_);
      if (closed_) return false;
      q_.push_back(std::move(v));
    }
    cv_.notify_one();              // 在锁外通知：被唤醒的线程不用马上又等锁
    return true;
  }
  std::optional<T> pop() {         // 队列为空时阻塞；关闭且取空后返回 nullopt
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
    cv_.notify_all();              // 叫醒所有在等的消费者，让它们看到"已关闭"
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<T> q_;                // 由 m_ 保护
  bool closed_ = false;            // 由 m_ 保护
};
```

一个生产者、两个消费者：

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
    q.close();                     // 生产完毕
  });
  std::vector<long> handled(2, 0), tokens(2, 0);   // 每个消费者只写自己的那一格：没有共享
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

```text title="输出"
处理了 100 个请求，共 550 个 token
```

`handled` 和 `tokens` 由两个线程同时写，但每个线程只写自己下标的元素——不同的内存位置，不构成数据竞争（它们挨得很近，有伪共享，统计量频繁更新时应该按缓存行隔开，见[对象布局、对齐与缓存](../memory/layout.md#伪共享)）。
主线程在 `join()` 之后再读，`join()` 建立了"线程里的写先于 join 返回"的关系，所以读是安全的。

## `std::jthread` 与协作式停止

后台线程（指标上报、心跳、KV 卸载）通常是一个循环，需要一种干净的方式让它退出。C++20 的 `std::jthread` 自带一个**停止令牌**：线程函数的第一个参数可以是 `std::stop_token`，外面调用 `request_stop()`（或者 `jthread` 析构时自动调用）后，`stop_requested()` 变为真：

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
        ticks.fetch_add(1);        // 假装在上报一次指标
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      std::printf("reporter 收到停止请求，退出\n");
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }   // jthread 析构：先 request_stop()，再 join()
  std::printf("上报过：%s\n", ticks.load() > 0 ? "是" : "否");
}
```

```text title="输出"
reporter 收到停止请求，退出
上报过：是
```

如果线程在条件变量上睡眠，用 `std::condition_variable_any` 的 `wait(lock, stop_token, 谓词)` 重载，停止请求会把它唤醒。

## 线程还是进程

C++ 服务里，同一进程的多个线程共享内存，通信最便宜；Python 服务受 GIL 限制，SGLang、vLLM 都把 tokenizer、调度器、detokenizer 拆成**多个进程**，用 ZMQ 传消息（见 mini-sglang 的[消息与 ZMQ](minisgl://serve/message/)）。
两种结构的共同点是**用消息传递代替共享状态**：每个组件独占自己的数据，只通过队列交换消息。这比在共享数据上精心加锁要容易做对得多。

## 练习

1. 给 `BlockingQueue` 加上容量上限：队列满时 `push` 阻塞，直到有空位或者队列被关闭（这叫**反压**：下游处理不过来时，让上游慢下来，而不是无限堆积请求）。写一个测试：容量 4，生产者快速放 20 个，消费者每取一个睡 1 毫秒，检查队列长度从未超过 4，且 20 个都被处理。要求在 TSan 下没有报告。

??? success "参考答案"
    需要两个条件变量："不空"给消费者等，"不满"给生产者等。`close()` 要同时唤醒两边：

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
        not_full_.notify_one();   // 空出了一个位置
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
          std::this_thread::sleep_for(std::chrono::milliseconds(1));   // 慢消费者
        }
      });
      for (int i = 0; i < 20; ++i) q.push(i);
      q.close();
      consumer.join();
      std::printf("处理 %d 个，队列长度从未超过 4：%s\n", handled, q.max_seen() <= 4 ? "是" : "否");
    }
    ```

    ```text title="输出"
    处理 20 个，队列长度从未超过 4：是
    ```

2. 下面的消费者代码有两个问题，指出来：

    ```cpp
    std::unique_lock lk(m);
    if (q.empty()) cv.wait(lk);
    auto item = q.front();
    q.pop_front();
    ```

??? success "参考答案"
    1. 用 `if` 只检查一次：虚假唤醒、或者被唤醒时元素已经被另一个消费者取走，`q.front()` 就会在空队列上调用（未定义行为）。要用 `cv.wait(lk, [&] { return !q.empty(); })` 或 `while (q.empty()) cv.wait(lk);`；
    2. 没有处理"关闭"：生产者结束后，消费者会永远睡在 `wait` 上，线程无法退出。谓词里要加上 `closed`，并在队列为空且已关闭时返回。

## 小结

- [x] 数据竞争是未定义行为；用 TSan（`-fsanitize=thread`）检查所有并发代码，"输出看起来对"不能证明没有竞争。
- [x] 锁保护数据：临界区要短，不在锁内做 I/O 和回调；多把锁用 `scoped_lock` 或固定顺序，避免死锁。
- [x] 条件变量配 `unique_lock`，`wait` 必须带谓词；关闭队列时 `notify_all` 让所有等待者退出。
- [x] 后台线程用 `std::jthread` 和 `stop_token` 协作式退出。
- [x] 最好的并发设计是减少共享：组件各自持有数据，通过队列传递消息。
