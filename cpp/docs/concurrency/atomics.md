# atomic 与内存序

<p class="lead">锁能解决大部分并发问题，但在最热的路径上——统计计数、"数据已就绪"的标志、无锁队列的读写指针、GPU 与 CPU 之间或卡与卡之间的信号——要用原子操作。原子操作本身不难，难的是<strong>内存序</strong>：一个线程写的数据，另一个线程什么时候、以什么顺序看到。这一章用最少的概念讲清楚 relaxed、acquire / release 和 seq_cst，并把它们和 GPU 通信库里的信号机制对应起来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个线程先写数据、再把 `ready` 设为 `true`；另一个线程看到 `ready == true` 后读数据。`ready` 是 `std::atomic<bool>` 就够了吗？用什么内存序？
    2. 请求计数器用什么内存序？为什么？
    3. `compare_exchange_weak` 为什么通常写在循环里？失败时它做了什么？
    4. 自旋锁的 `lock()` 为什么是 acquire、`unlock()` 为什么是 release？
    5. GPU 上一个 kernel 写完一块数据后要设置一个标志通知另一张卡，中间为什么要一个 fence？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 原子性够了，但顺序不够：还要保证"读到 `ready == true` 时一定能看到写好的数据"。写方用 `ready.store(true, release)`，读方用 `ready.load(acquire)`（或者都用默认的 `seq_cst`）。
    2. `relaxed`：计数器只需要原子性，不用来同步其他数据；最终读取发生在 `join()` 之后，已经有同步保证。
    3. 它允许"伪失败"（值没变也可能返回失败），而且并发时别的线程可能改了值，所以要在循环里重试。失败时它把 `expected` 更新成当前的实际值，下一轮直接用它计算新值。
    4. `lock()` 用 acquire：获得锁之后，要能看到上一个持有者在临界区里写的所有数据；`unlock()` 用 release：临界区里的写入要在释放锁之前对下一个获得锁的线程可见。
    5. GPU 的写入可能被重排或者还停留在缓存里：没有 fence，对方可能先看到标志、后看到数据。`__threadfence_system()`（或者带 release 语义的写）保证数据先对其他设备可见，再写标志——和 CPU 上的 release / acquire 是同一回事。

## 原子操作

`std::atomic<T>` 上的每个操作都是不可分割的：不会出现"读到一半被别人写了"，多个线程同时对它 `fetch_add` 也不会丢失更新。常用操作：

| 操作 | 含义 |
| --- | --- |
| `load()` / `store(v)` | 读 / 写 |
| `fetch_add(d)` / `fetch_sub(d)` | 加减，返回旧值 |
| `exchange(v)` | 写入新值，返回旧值 |
| `compare_exchange_weak/strong(expected, desired)` | 如果当前值等于 `expected` 就改成 `desired` 并返回 `true`；否则把当前值写回 `expected`、返回 `false`（CAS） |
| `wait(old)` / `notify_one()` / `notify_all()` | C++20：阻塞等待值不再等于 `old`（不用条件变量） |

`int`、`long`、指针这些大小的原子类型在主流平台上都是**无锁**的（`std::atomic<T>::is_always_lock_free`），直接编译成一条带 `lock` 前缀的指令或普通的读写指令。

## 为什么还需要内存序

原子性只保证**这一个变量**的操作不被撕裂。但编译器和 CPU 都会重排指令：只要单线程的结果不变，写数据和写标志的顺序就可能被调换，另一个线程于是可能先看到标志、后看到数据。

下面这个"发布配置"的例子里，`ready` 是原子的，但用了 `relaxed`（只保证原子性、不保证顺序）。**这个程序有 bug：**

```cpp title="publish_relaxed.cpp" sanitize="thread" expect="fail"
#include <atomic>
#include <cstdio>
#include <thread>

struct Config {
  int max_batch;
  int page_size;
};
Config config;                    // 普通变量
std::atomic<bool> ready{false};

int main() {
  std::thread loader([] {
    config = {256, 16};
    ready.store(true, std::memory_order_relaxed);   // 只保证原子，不保证 config 先写完
  });
  std::thread worker([] {
    while (!ready.load(std::memory_order_relaxed)) {
    }
    std::printf("%d\n", config.max_batch);           // 可能读到旧值：数据竞争
  });
  loader.join();
  worker.join();
}
```

```text title="ThreadSanitizer 的报告（节选）"
WARNING: ThreadSanitizer: data race (pid=12345)
  Read of size 4 at 0x55... by thread T2:
    #0 main::{lambda()#2}::operator()() const publish_relaxed.cpp:20
  Previous write of size 8 at 0x55... by thread T1:
    #0 main::{lambda()#1}::operator()() const publish_relaxed.cpp:14
```

在 x86 上这个程序"几乎总能"跑对（x86 的硬件内存模型比较强），在 ARM 服务器（Grace、鲲鹏、Graviton）上就可能真的读到旧值——而编译器在任何平台上都有权重排。

## acquire / release：发布与获取

修法是用 **release 写**和 **acquire 读**配对：

- `store(v, memory_order_release)`：这条写**之前**的所有读写，不会被重排到它之后；
- `load(memory_order_acquire)`：这条读**之后**的所有读写，不会被重排到它之前；
- 当 acquire 读**读到了** release 写的值，写线程在 release 之前做的所有事，对读线程在 acquire 之后都可见。这叫 release 写"同步于"acquire 读。

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
    ready.store(true, std::memory_order_release);    // 之前的写（config）都"发布"出去
  });
  std::thread worker([] {
    while (!ready.load(std::memory_order_acquire)) { // 读到 true 之后，config 一定是新的
    }
    std::printf("max_batch=%d page_size=%d\n", config.max_batch, config.page_size);
  });
  loader.join();
  worker.join();
}
```

```text title="输出"
max_batch=256 page_size=16
```

这是最重要的一种模式——**先写数据，再用 release 设标志；用 acquire 读到标志，再读数据**。无锁队列、"权重加载完成"的通知、跨线程交接一个指针，都是它。

## 该用哪种内存序

| 内存序 | 保证 | 典型用途 |
| --- | --- | --- |
| `relaxed` | 只保证这个变量上的操作是原子的 | 统计计数器（请求数、token 数）、只关心最终值的累加、ID 生成器 |
| `acquire` / `release` | 成对使用，建立"先于"关系 | 发布数据（标志 + 数据）、锁、无锁队列的读写指针 |
| `seq_cst`（默认） | 所有 seq_cst 操作有一个全局一致的顺序 | 拿不准的时候；多个标志之间需要全局顺序的算法 |

几点建议：

- 不写内存序时默认是 `seq_cst`，永远是正确的，只是可能慢一点（x86 上 seq_cst 的 store 要用 `xchg` 或加 `mfence`）。**先写对，确认是瓶颈再放松**；
- 计数器用 `relaxed` 是安全的，前提是没有人依赖"计数器变了"来推断别的数据已经就绪；
- 所有用了非默认内存序的地方，都应该写一行注释说明它和谁配对。

## CAS 循环

`compare_exchange` 是构建无锁算法的基本工具：读出当前值，算出新值，**只有在这期间没人改过时**才写入，否则用最新的值重试。比如并发地更新"峰值显存占用"：

```cpp title="atomic_max.cpp" sanitize="thread"
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

void atomic_max(std::atomic<long>& target, long v) {
  long cur = target.load(std::memory_order_relaxed);
  while (cur < v && !target.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    // 失败时 cur 已经被更新成当前值，循环回去重新比较
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

```text title="输出"
峰值 = 9972
```

`weak` 版本允许"虚假失败"（值明明相等也返回 `false`，在 ARM 这类用 LL/SC 实现 CAS 的平台上更高效），所以总是写在循环里；不在循环里时用 `strong`。

!!! warning "ABA 问题"
    CAS 只比较值。如果一个线程读到 `A`，期间别的线程把它改成 `B` 又改回 `A`，CAS 仍然会成功，但中间的变化被忽略了。
    用指针做无锁栈时，这意味着"同一个地址的节点被释放后又被重新分配"——经典的 ABA 问题。解决办法是给指针配一个版本号一起 CAS，或者用危险指针、基于 epoch 的回收。实践中，能用单生产者单消费者的环形队列（下一章）就不要自己写多生产者的无锁链表结构。

## 自旋锁

用一个原子标志就能实现最简单的锁：`lock()` 用 acquire 把标志从 `false` 改成 `true`，`unlock()` 用 release 改回来。acquire / release 正好保证临界区里的读写不会"漏出"锁之外：

```cpp title="spinlock.cpp" sanitize="thread"
#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

class SpinLock {
 public:
  void lock() {
    while (flag_.test_and_set(std::memory_order_acquire)) {   // 抢锁
      while (flag_.test(std::memory_order_relaxed)) {         // 抢不到时只读不写，避免缓存行来回失效
      }
    }
  }
  void unlock() { flag_.clear(std::memory_order_release); }

 private:
  std::atomic_flag flag_;   // C++20 起默认初始化为"未设置"
};

int main() {
  SpinLock lock;
  long counter = 0;
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < 10000; ++i) {
        std::lock_guard g(lock);   // 任何有 lock()/unlock() 的类型都能配 lock_guard
        ++counter;
      }
    });
  }
  for (auto& th : ts) th.join();
  std::printf("counter = %ld\n", counter);
}
```

```text title="输出"
counter = 40000
```

自旋锁只适合**临界区极短、且线程数不超过核数**的场景：拿不到锁的线程一直占着 CPU 空转，持锁线程被操作系统切走时，其他线程会白白转完整个时间片。一般情况用 `std::mutex`（现代实现会先自旋一小会儿再睡眠）。

## C++20 的原子等待

需要"等一个标志变化"又不想空转时，C++20 的 `wait` / `notify` 直接在原子变量上睡眠（Linux 上是 futex），比条件变量轻量：

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
      weights_loaded.wait(false, std::memory_order_acquire);   // 值还是 false 就睡眠
      started.fetch_add(1, std::memory_order_relaxed);
    });
  }
  weights_loaded.store(true, std::memory_order_release);        // 加载完成
  weights_loaded.notify_all();
  for (auto& w : workers) w.join();
  std::printf("%d 个工作线程都开始了\n", started.load());
}
```

```text title="输出"
3 个工作线程都开始了
```

## 和 GPU 的对应

同样的问题在 GPU 上一模一样，只是范围更大：

- 一个 block 写共享内存、另一个线程读，要 `__syncthreads()`；
- 一个 kernel 写全局内存、再用原子操作设置标志通知**别的 block**，写数据和写标志之间要 `__threadfence()`（或者用带 release 语义的原子操作），读的一方要 acquire——这正是 CUDA 手册里[单遍归约](cuda://kernels/reduction/)用"最后一个完成的 block 汇总"时要处理的问题；
- 跨卡的一方通知另一方（NVSHMEM 的 `put` + `signal`、DeepEP 的低延迟模式）：先把数据写到对端，再写一个信号；对端轮询信号，读到后才读数据。中间的 fence 就是 release / acquire 在跨设备场景下的版本。

读这些代码时，只要找到"数据写在哪、标志写在哪、它们之间用什么保证顺序"，逻辑就清楚了。

!!! interview "面试怎么答"
    内存序题的核心模式：生产者先写数据、再用 release 写标志；消费者用 acquire 读到标志后再读数据，就一定看得到数据——只把标志换成 `atomic` 而用 `relaxed` 不够。计数器只要原子性、不需要排序，用 `relaxed`；拿不准就用默认的 `seq_cst`，再用 TSan 验证。CAS 写在循环里：`compare_exchange_weak` 可能伪失败，失败时把当前值写回 `expected`；多生产者的无锁结构要当心 ABA。自旋锁的 `lock` 用 acquire、`unlock` 用 release，保证临界区里的读写不会被移到锁外。GPU 上的 `__threadfence`、跨卡通信的 put + signal 是同一套规则在更大范围上的应用。

## 练习

1. 为下面的场景选择内存序，并说明理由：
    1. 所有线程累加"已处理的 token 总数"，只在最后打印；
    2. 后台线程加载完一个 LoRA 适配器后，把指向它的指针存进 `std::atomic<Adapter*>`，请求线程读到非空指针后直接使用它；
    3. 主线程设置 `std::atomic<bool> stop`，工作线程在循环里检查它，看到 `true` 就退出（退出前不需要读取主线程写的其他数据）。

??? success "参考答案"
    1. `relaxed`：只需要原子性，最终读取发生在 `join()` 之后，`join()` 已经提供了同步。
    2. 存指针用 `release`，读指针用 `acquire`：请求线程要看到适配器对象被完整构造出来之后的内容，这是典型的"发布"。只用 `relaxed` 可能读到一个"指针已经可见、但对象内容还没写完"的状态。
    3. `relaxed` 就够了：标志本身不携带其他数据；线程最迟在下一次检查时看到它。如果退出前要读取主线程在设置 `stop` 之前写的数据（比如"退出原因"），就要 release / acquire。

2. 用 `std::atomic<int>` 实现一个只用一次的"门闩"：`count_down()` 被调用 N 次后，所有 `wait()` 的线程都被放行。（C++20 已经有 `std::latch`，这里自己实现一遍。）要求在 TSan 下没有报告。

??? success "参考答案"
    ```cpp title="my_latch.cpp" sanitize="thread"
    #include <atomic>
    #include <cstdio>
    #include <thread>
    #include <vector>

    class Latch {
     public:
      explicit Latch(int n) : count_(n) {}
      void count_down() {
        if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) count_.notify_all();   // 最后一个到达的负责唤醒
      }
      void wait() {
        int c = count_.load(std::memory_order_acquire);
        while (c != 0) {
          count_.wait(c, std::memory_order_acquire);   // 值还是 c 就睡眠
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
          loaded[i] = 100 + i;   // 各自加载一个权重分片
          ready.count_down();
        });
      }
      ready.wait();               // 所有分片都加载完才开始服务
      int sum = 0;
      for (int v : loaded) sum += v;
      std::printf("所有分片就绪，校验和 %d\n", sum);
      for (auto& t : shards) t.join();
    }
    ```

    ```text title="输出"
    所有分片就绪，校验和 406
    ```

    `fetch_sub` 用 `acq_rel`：release 让每个线程在 `count_down` 之前写的 `loaded[i]` 被发布出去，acquire 让最后一个线程能看到之前所有线程的写入；`wait()` 用 acquire 读到 0 之后，主线程就能安全地读 `loaded`。

## 小结

- [x] 原子性只保证单个变量不被撕裂；跨变量的可见顺序要靠内存序。
- [x] "先写数据、再 release 写标志；acquire 读到标志、再读数据"是最核心的模式，锁、无锁队列、跨线程交接指针都建立在它上面。
- [x] 计数器用 `relaxed`，发布数据用 acquire / release，拿不准就用默认的 `seq_cst`，并用 TSan 验证。
- [x] CAS 写在循环里，失败时 `expected` 被更新成当前值；自己写多生产者无锁结构要当心 ABA。
- [x] GPU 上的 `__threadfence`、跨卡通信的 put + signal，是同一套规则在更大范围上的应用。
