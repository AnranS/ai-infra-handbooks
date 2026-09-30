# 对象布局、对齐与缓存

<p class="lead">CPU 和 GPU 读内存都不是一个字节一个字节地读，而是一整行一整行地读：CPU 是 64 字节的缓存行，GPU 是 32 字节的扇区。数据怎么排布，决定了每读进来的一行里有多少是有用的。这一章讲对象在内存里的样子、对齐的规则，以及怎样写出对缓存友好的数据结构——它和 CUDA 里的"合并访存"是同一件事。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `struct { bool a; double b; int c; }` 的 `sizeof` 是多少？怎样重排字段让它变小？
    2. 为什么锁页内存、RDMA 注册的内存通常要按 4096 字节对齐？
    3. 调度器每一步只读所有请求的 `output_len` 字段，数组的结构体（AoS）和结构体的数组（SoA）哪个快？快多少？
    4. 什么是伪共享（false sharing）？怎么避免？
    5. 按列遍历一个行主序的大矩阵为什么慢？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 24 字节：`a` 之后填 7 字节，让 `b` 按 8 对齐，`c` 之后再填 4 字节，让总大小是 8 的倍数。按对齐从大到小排成 `double b; int c; bool a;` 是 16 字节。
    2. 操作系统按页（通常 4096 字节）管理和锁定物理内存，DMA、RDMA 注册和 `O_DIRECT` 也以页为单位；地址和长度按页对齐，才能直接锁页、注册，不会跨页带上无关的数据。
    3. SoA 快得多：只读一个字段时，SoA 的每个缓存行里全是有用的 `output_len`，AoS 每个缓存行只有一小部分是有用的。本章实测约 10 倍（对应每个缓存行里有用字节之比，SoA 还能向量化）。
    4. 两个线程频繁写的不同变量恰好在同一个缓存行里，每次写都要让对方核上的这一行失效，缓存行在核之间来回传递，性能大幅下降。把它们按 64 字节隔开（`alignas(64)` 或填充）。
    5. 行主序时一行的元素是连续的，按列遍历每次访问都跳过一整行，每个缓存行只用到一个元素，还会不断被换出；遍历顺序要和存储顺序一致。

## `sizeof`、`alignof` 与填充

点下面的字段把它往前挪，或者直接按"按对齐从大到小排"，看 `sizeof` 怎么变：

<div class="aig-widget" data-widget="structlayout"></div>

每种类型都有一个**对齐要求**（`alignof`）：它的地址必须是这个数的倍数。`int32_t` 是 4，`double` 和指针是 8。结构体的对齐要求是其成员中最大的那个；为了让每个成员都对齐，编译器会在成员之间插入**填充字节**，结构体的总大小也会补齐到对齐要求的倍数：

```cpp title="layout.cpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>

struct Bad {                  // 字段顺序随手写
  bool done;
  double temperature;
  std::int32_t len;
  bool stream;
  std::int64_t id;
};

struct Good {                 // 按对齐要求从大到小排
  double temperature;
  std::int64_t id;
  std::int32_t len;
  bool done;
  bool stream;
};

int main() {
  std::printf("Bad：sizeof=%zu alignof=%zu\n", sizeof(Bad), alignof(Bad));
  std::printf("Good：sizeof=%zu alignof=%zu\n", sizeof(Good), alignof(Good));
  std::printf("Bad 的字段偏移：done=%zu temperature=%zu len=%zu stream=%zu id=%zu\n", offsetof(Bad, done),
              offsetof(Bad, temperature), offsetof(Bad, len), offsetof(Bad, stream), offsetof(Bad, id));
}
```

```text title="输出"
Bad：sizeof=32 alignof=8
Good：sizeof=24 alignof=8
Bad 的字段偏移：done=0 temperature=8 len=16 stream=20 id=24
```

`Bad` 在 `done` 后面填了 7 个字节才能让 `temperature` 从 8 的倍数开始，在 `stream` 后面又填了 3 个字节。**把字段按对齐要求从大到小排**，通常就能消掉大部分填充。
一个请求的元数据少 8 字节看起来不多，但放在一个百万级的数组里、或者要通过网络和 GPU 传输时，就是 25% 的带宽。

需要和别的程序（Python 的 `struct`、GPU kernel、网络协议）共享二进制布局时，用 `static_assert(sizeof(T) == ...)` 和 `static_assert(offsetof(T, f) == ...)` 把布局钉死，改动时编译期就能发现。

## 对齐：`alignas` 与对齐的分配

有时需要比类型本身更大的对齐：

- **向量化访问**：一次读 16 字节（CPU 的 SSE、GPU 的 `float4` / 128 位加载）要求地址按 16 字节对齐；
- **页对齐**：`cudaHostRegister` 锁页、RDMA 注册内存、`O_DIRECT` 读权重文件，都要求地址和长度按页（通常 4096 字节）对齐；
- **缓存行对齐**：避免伪共享（见下文）。

```cpp title="aligned.cpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct alignas(16) Float4 {
  float x, y, z, w;
};
struct alignas(64) PaddedCounter {   // 独占一个缓存行
  std::int64_t value;
};

bool aligned_to(const void* p, std::size_t a) { return reinterpret_cast<std::uintptr_t>(p) % a == 0; }

int main() {
  std::printf("alignof(Float4)=%zu sizeof(PaddedCounter)=%zu\n", alignof(Float4), sizeof(PaddedCounter));
  void* page = std::aligned_alloc(4096, 1 << 20);   // 大小必须是对齐值的倍数
  std::printf("aligned_alloc 按 4096 对齐：%s\n", aligned_to(page, 4096) ? "是" : "否");
  std::free(page);
  auto* counters = new PaddedCounter[4];            // C++17 起 new 会遵守 alignas
  std::printf("new PaddedCounter[] 按 64 对齐：%s\n", aligned_to(counters, 64) ? "是" : "否");
  delete[] counters;
}
```

```text title="输出"
alignof(Float4)=16 sizeof(PaddedCounter)=64
aligned_alloc 按 4096 对齐：是
new PaddedCounter[] 按 64 对齐：是
```

`std::vector<float>` 的缓冲区只保证按 `alignof(std::max_align_t)`（通常 16）对齐；需要更大的对齐时，用 `aligned_alloc` 配合 RAII 包装，或者给 `vector` 一个对齐分配器（见[分配器与内存池](allocators.md)）。

## 缓存：数据怎么被读进来

CPU 访问内存时，每次把一整个 **64 字节的缓存行**搬进缓存。粗略的延迟量级：

| 层级 | 容量（每核 / 共享） | 延迟 |
| --- | --- | --- |
| L1 | 几十 KB | 约 1 ns（4～5 个周期） |
| L2 | 1～2 MB | 约 4 ns |
| L3 | 几十 MB（共享） | 约 10～20 ns |
| 内存 | — | 约 80～100 ns |

所以决定性能的往往不是"做了多少次运算"，而是"读进来的缓存行里有多少字节是有用的"。这和 GPU 上的合并访存完全是一回事——GPU 以 32 字节的扇区为单位读显存，一个 warp 的 32 个线程如果读的是连续地址，几个扇区就够了（见 CUDA 手册的[内存层次](cuda://basics/memory/)）。

### 数组的结构体还是结构体的数组

调度器里通常有一个"所有请求的元数据"表。每一步它往往只关心其中一两个字段：统计已经生成的 token 数、找出已经结束的请求、计算每个请求需要的块数。

- **AoS**（Array of Structs）：`std::vector<Request>`，一个请求的所有字段挨在一起；
- **SoA**（Struct of Arrays）：每个字段一个数组，同一个字段的所有请求挨在一起。

只读一个字段时，AoS 每读进一个 64 字节的缓存行，只有 4 个字节有用；SoA 读进来的每个字节都有用：

```cpp title="aos_soa.cpp" sanitize="none" flags="-O2"
// g++ -std=c++20 -O2 aos_soa.cpp && ./a.out
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

struct RequestAoS {   // 一个请求的所有字段放在一起：64 字节
  std::int64_t id;
  std::int32_t prompt_len, output_len;
  float temperature, top_p;
  std::int32_t num_blocks, priority;
  bool finished, streaming;
  char padding[30];
};

struct RequestsSoA {  // 每个字段一个数组
  std::vector<std::int64_t> id;
  std::vector<std::int32_t> prompt_len, output_len;
};

template <class F>
double best_ms(F&& f) {
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  return best;
}

volatile std::int64_t sink;

int main() {
  const int n = 4'000'000;
  std::vector<RequestAoS> aos(n);
  RequestsSoA soa;
  soa.output_len.resize(n);
  for (int i = 0; i < n; ++i) aos[i].output_len = soa.output_len[i] = i % 100;
  // 调度器每一步只关心一个字段：统计所有请求已经生成的 token 数
  double t_aos = best_ms([&] { std::int64_t s = 0; for (const auto& r : aos) s += r.output_len; sink = s; });
  double t_soa = best_ms([&] { std::int64_t s = 0; for (int v : soa.output_len) s += v; sink = s; });
  std::printf("sizeof(RequestAoS)=%zu\n", sizeof(RequestAoS));
  std::printf("只读一个字段：AoS %.2f ms，SoA %.2f ms\n", t_aos, t_soa);
}
```

```text title="一次运行的结果（x86 服务器）"
sizeof(RequestAoS)=64
只读一个字段：AoS 20.58 ms，SoA 2.17 ms
```

差了约 10 倍，正好对应"每个缓存行里有用的字节"之比（64 / 4 = 16，SoA 还能被向量化）。反过来，如果每次都要读一个请求的**全部**字段，AoS 更好：一个缓存行就拿到了所有东西。
推理引擎的"批次元数据"（每个请求的序列长度、块表、采样参数）大多组织成 SoA，因为它们最终要变成传给 GPU kernel 的一个个连续数组（`seq_lens`、`block_tables`、`temperatures`）。

### 访问顺序

同样的数据，访问顺序不同，性能也天差地别。行主序存储的矩阵按列遍历，每次访问都跨过一整行，每个缓存行只用到 4 个字节：

```cpp title="traverse.cpp" sanitize="none" flags="-O2"
// g++ -std=c++20 -O2 traverse.cpp && ./a.out
#include <chrono>
#include <cstdio>
#include <vector>

volatile float sink;

int main() {
  const int n = 4096;
  std::vector<float> m(std::size_t(n) * n, 1.0f);   // 行主序：m[i * n + j]
  auto time = [&](bool by_row) {
    auto t0 = std::chrono::steady_clock::now();
    float s = 0;
    for (int a = 0; a < n; ++a)
      for (int b = 0; b < n; ++b) s += by_row ? m[std::size_t(a) * n + b] : m[std::size_t(b) * n + a];
    sink = s;
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  time(true);   // 预热
  std::printf("按行遍历 %.0f ms，按列遍历 %.0f ms\n", time(true), time(false));
}
```

```text title="一次运行的结果（x86 服务器）"
按行遍历 23 ms，按列遍历 175 ms
```

KV Cache 的布局就是这个问题：`[token][head][dim]` 还是 `[head][token][dim]`，取决于 kernel 最常沿哪个维度连续读。分页 KV 把一个块（比如 16 个 token）的数据放在一起，就是为了让 decode 时按块读 KV 仍然是连续的大块访问。

## 伪共享

多核之间通过缓存一致性协议（MESI）保持缓存行的一致：一个核写了某个缓存行，其他核里这一行的副本就失效了。如果两个线程写的是**不同的变量**，但这两个变量恰好在**同一个缓存行**里，这一行就会在两个核之间来回失效、搬运——这叫**伪共享**：

```cpp title="false_sharing.cpp" sanitize="none" flags="-O2"
// g++ -std=c++20 -O2 -pthread false_sharing.cpp && ./a.out
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

struct Plain {
  std::atomic<long> value{0};
};
struct alignas(64) Padded {   // 每个计数器独占一个 64 字节的缓存行
  std::atomic<long> value{0};
};

template <class Counter>
double run(int threads, long iters) {
  std::vector<Counter> counters(threads);
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; ++t)
    ts.emplace_back([&, t] {
      for (long i = 0; i < iters; ++i) counters[t].value.fetch_add(1, std::memory_order_relaxed);
    });
  for (auto& th : ts) th.join();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
  const int threads = 4;
  const long iters = 20'000'000;
  std::printf("sizeof(Plain)=%zu sizeof(Padded)=%zu\n", sizeof(Plain), sizeof(Padded));
  std::printf("%d 个线程各加自己的计数器 %ld 次：挨在一起 %.0f ms，按缓存行隔开 %.0f ms\n", threads, iters,
              run<Plain>(threads, iters), run<Padded>(threads, iters));
}
```

```text title="一次运行的结果（x86 服务器）"
sizeof(Plain)=8 sizeof(Padded)=64
4 个线程各加自己的计数器 20000000 次：挨在一起 1864 ms，按缓存行隔开 137 ms
```

四个线程之间没有任何逻辑上的共享，只因为计数器挨在一起就慢了一个数量级。每个线程各自维护的统计量、无锁队列的读指针和写指针、线程池里每个工作线程的状态，都应该按缓存行隔开。
C++17 提供了 `std::hardware_destructive_interference_size` 表示这个距离，但它的值依赖编译选项（g++ 使用时会给出警告），实践中通常直接写 64（部分 ARM 服务器是 128）。

!!! interview "面试怎么答"
    布局与缓存题：结构体的大小包含对齐填充，`{bool; double; int}` 是 24 字节，按对齐从大到小排成 `{double; int; bool}` 是 16 字节；跨语言、跨设备共享的布局用 `static_assert` 钉死大小和偏移。锁页内存、RDMA 注册、`O_DIRECT` 要按页对齐。性能取决于每个缓存行里有多少有用的字节：只读少数字段时结构体的数组（SoA）比数组的结构体（AoS）快一个数量级，遍历顺序要和存储顺序一致——这和 GPU 的合并访存是同一回事。不同线程频繁写的变量要按缓存行（64 字节）隔开，避免伪共享。

## 练习

1. 下面是一个请求在调度器里的状态。重排字段让它尽量小，写出重排后的 `sizeof`，并用 `static_assert` 固定下来。

    ```cpp
    struct SeqState {
      bool is_prefill;
      std::int64_t request_id;
      std::int16_t lora_slot;
      float temperature;
      bool finished;
      std::int32_t num_computed_tokens;
      double arrival_time;
      std::int8_t priority;
    };
    ```

??? success "参考答案"
    原来的布局：`is_prefill` 后填 7 字节、`lora_slot` 后填 2 字节、`finished` 后填 3 字节、`priority` 后填 7 字节，一共 48 字节。按对齐从大到小排：两个 8 字节、两个 4 字节、一个 2 字节、三个 1 字节，正好 $16 + 8 + 2 + 3 = 29$，补齐到 32：

    ```cpp title="seq_state.cpp"
    #include <cstdint>
    #include <cstdio>

    struct SeqStateBad {
      bool is_prefill;
      std::int64_t request_id;
      std::int16_t lora_slot;
      float temperature;
      bool finished;
      std::int32_t num_computed_tokens;
      double arrival_time;
      std::int8_t priority;
    };

    struct SeqState {
      std::int64_t request_id;
      double arrival_time;
      float temperature;
      std::int32_t num_computed_tokens;
      std::int16_t lora_slot;
      bool is_prefill;
      bool finished;
      std::int8_t priority;
    };
    static_assert(sizeof(SeqState) == 32, "布局变了：检查字段顺序");

    int main() { std::printf("重排前 %zu 字节，重排后 %zu 字节\n", sizeof(SeqStateBad), sizeof(SeqState)); }
    ```

    ```text title="输出"
    重排前 48 字节，重排后 32 字节
    ```

2. 一层的 KV Cache 按 `[num_blocks][block_size=16][num_kv_heads=8][head_dim=128]` 的 bf16 存储。decode kernel 为**一个 KV 头**读取某个块的全部 16 个 token 的 K：每个 token 读到的连续字节有多少？一共读了多少个 64 字节的缓存行（按 CPU 的缓存行算）？如果布局改成 `[num_blocks][num_kv_heads][block_size][head_dim]` 呢？

??? success "参考答案"
    原布局：一个 token 的一个头是 $128 \times 2 = 256$ 字节连续，正好 4 个缓存行；相邻 token 之间隔着 $8 \times 256 = 2048$ 字节。16 个 token 共 $16 \times 4 = 64$ 个缓存行，但分散成 16 段。
    改成"头在块内的外层"之后，同一个头的 16 个 token 连成一段 $16 \times 256 = 4096$ 字节，还是 64 个缓存行，但是一段连续的访问，硬件预取和 GPU 的大块加载（TMA、`cp.async`）都更高效。
    vLLM 的分页 KV 使用 `[num_blocks, block_size, num_kv_heads, head_size]` 还是 `[num_blocks, num_kv_heads, block_size, head_size]`，在不同的注意力后端里就是按这个考虑选的（见推理系统手册的[分页 KV Cache](serving://engine/paged-kv/)）。

## 小结

- [x] 结构体的大小包含对齐填充；字段按对齐要求从大到小排能消掉大部分填充；跨语言、跨设备共享的布局用 `static_assert` 钉死。
- [x] 向量化访问要 16 字节对齐，锁页、RDMA、`O_DIRECT` 要页对齐；用 `alignas`、`aligned_alloc` 或对齐分配器。
- [x] 性能取决于每个缓存行里有多少有用的字节：只读少数字段时 SoA 比 AoS 快一个数量级，遍历顺序要和存储顺序一致。这和 GPU 的合并访存是同一件事。
- [x] 不同线程频繁写的变量要按缓存行（64 字节）隔开，避免伪共享。
