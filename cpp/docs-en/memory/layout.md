# Object layout, alignment and the cache

<p class="lead">Neither a CPU nor a GPU reads memory a byte at a time: it reads a whole line at a time, a 64-byte cache line on a CPU and a 32-byte sector on a GPU. How the data is laid out decides how much of each line brought in is useful. This chapter covers what an object looks like in memory, the alignment rules, and how to write cache-friendly data structures, which is the same thing as CUDA's "coalesced access".</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is `sizeof` for `struct { bool a; double b; int c; }`? How do you reorder the fields to make it smaller?
    2. Why do pinned memory and memory registered with RDMA usually have to be aligned to 4096 bytes?
    3. The scheduler reads only every request's `output_len` field each step. Which is faster, an array of structs (AoS) or a struct of arrays (SoA)? By how much?
    4. What is false sharing? How do you avoid it?
    5. Why is traversing a large row-major matrix by column slow?

??? success "Answers (try it yourself first, then expand)"
    1. 24 bytes: 7 bytes of padding after `a` to align `b` to 8, and 4 more after `c` to make the total a multiple of 8. Ordered by alignment, largest first, `double b; int c; bool a;` is 16 bytes.
    2. The operating system manages and locks physical memory in pages (usually 4096 bytes), and DMA, RDMA registration and `O_DIRECT` all work in pages; only an address and length aligned to a page can be pinned or registered directly without dragging unrelated data from a neighbouring page.
    3. SoA is far faster: reading one field, every cache line of SoA is full of useful `output_len`s, while only a small part of each AoS line is useful. Measured in this chapter at about 10 times (matching the ratio of useful bytes per cache line, and SoA vectorizes besides).
    4. Two threads frequently write different variables that happen to sit in the same cache line, so every write invalidates that line on the other core and the line bounces between cores, with a large loss of performance. Separate them by 64 bytes (`alignas(64)` or padding).
    5. In row-major storage a row's elements are contiguous, so traversing by column skips a whole row on each access, uses one element per cache line and keeps evicting them; the traversal order has to match the storage order.

## `sizeof`, `alignof` and padding {#sizeofalignof-与填充}

Click a field below to move it forward, or just take "ordered by alignment, largest first", and watch `sizeof` change:

<div class="aig-widget" data-widget="structlayout"></div>

Every type has an **alignment requirement** (`alignof`): its address has to be a multiple of that number. `int32_t` is 4, and `double` and pointers are 8. A struct's alignment is the largest of its members'; to align every member the compiler inserts **padding bytes** between them, and the struct's total size is rounded up to a multiple of its alignment:

```cpp title="layout.cpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>

struct Bad {                  // the fields written in no particular order
  bool done;
  double temperature;
  std::int32_t len;
  bool stream;
  std::int64_t id;
};

struct Good {                 // ordered by alignment, largest first
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

```text title="output"
Bad：sizeof=32 alignof=8
Good：sizeof=24 alignof=8
Bad 的字段偏移：done=0 temperature=8 len=16 stream=20 id=24
```

`Bad` pads 7 bytes after `done` so that `temperature` starts at a multiple of 8, and 3 more after `stream`. **Ordering the fields by alignment, largest first**, usually removes most of the padding.
Saving 8 bytes in one request's metadata does not sound like much, but in an array of a million, or when it is sent over the network and to the GPU, that is 25% of the bandwidth.

When the binary layout is shared with something else (Python's `struct`, a GPU kernel, a network protocol), nail it down with `static_assert(sizeof(T) == ...)` and `static_assert(offsetof(T, f) == ...)` so a change is caught at compile time.

## Alignment: `alignas` and aligned allocation {#对齐alignas-与对齐的分配}

Sometimes an alignment larger than the type's own is needed:

- **vectorized access**: reading 16 bytes at once (a CPU's SSE, a GPU's `float4` / 128-bit load) requires the address to be 16-byte aligned;
- **page alignment**: `cudaHostRegister` pinning, registering memory with RDMA and reading a weight file with `O_DIRECT` all require the address and the length to be page-aligned (usually 4096 bytes);
- **cache line alignment**: to avoid false sharing (below).

```cpp title="aligned.cpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct alignas(16) Float4 {
  float x, y, z, w;
};
struct alignas(64) PaddedCounter {   // a cache line to itself
  std::int64_t value;
};

bool aligned_to(const void* p, std::size_t a) { return reinterpret_cast<std::uintptr_t>(p) % a == 0; }

int main() {
  std::printf("alignof(Float4)=%zu sizeof(PaddedCounter)=%zu\n", alignof(Float4), sizeof(PaddedCounter));
  void* page = std::aligned_alloc(4096, 1 << 20);   // the size has to be a multiple of the alignment
  std::printf("aligned_alloc 按 4096 对齐：%s\n", aligned_to(page, 4096) ? "是" : "否");
  std::free(page);
  auto* counters = new PaddedCounter[4];            // from C++17, new respects alignas
  std::printf("new PaddedCounter[] 按 64 对齐：%s\n", aligned_to(counters, 64) ? "是" : "否");
  delete[] counters;
}
```

```text title="output"
alignof(Float4)=16 sizeof(PaddedCounter)=64
aligned_alloc 按 4096 对齐：是
new PaddedCounter[] 按 64 对齐：是
```

A `std::vector<float>`'s buffer is only guaranteed to be aligned to `alignof(std::max_align_t)` (usually 16); for a larger alignment, use `aligned_alloc` with an RAII wrapper, or give the `vector` an aligned allocator (see [allocators and memory pools](allocators.md)).

## The cache: how data is brought in {#缓存数据怎么被读进来}

Every time a CPU touches memory it brings a whole **64-byte cache line** into the cache. Roughly, the latencies:

| Level | Capacity (per core / shared) | Latency |
| --- | --- | --- |
| L1 | tens of KB | about 1 ns (4-5 cycles) |
| L2 | 1-2 MB | about 4 ns |
| L3 | tens of MB (shared) | about 10-20 ns |
| memory | — | about 80-100 ns |

So what decides performance is often not "how many operations were done" but "how many bytes of each cache line brought in were useful". This is exactly coalesced access on a GPU, which reads device memory in 32-byte sectors, so a warp's 32 threads reading contiguous addresses need only a few sectors (see the CUDA handbook's [memory hierarchy](cuda://basics/memory/)).

### An array of structs or a struct of arrays {#数组的结构体还是结构体的数组}

A scheduler usually has a table of "every request's metadata". Each step it often cares about only one or two fields: counting the tokens generated, finding the requests that have finished, computing how many blocks each request needs.

- **AoS** (Array of Structs): `std::vector<Request>`, one request's fields next to each other;
- **SoA** (Struct of Arrays): one array per field, all the requests' values of a field next to each other.

Reading one field, AoS brings in a 64-byte cache line of which 4 bytes are useful; every byte SoA brings in is useful:

```cpp title="aos_soa.cpp" sanitize="none" flags="-O2"
// g++ -std=c++20 -O2 aos_soa.cpp && ./a.out
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

struct RequestAoS {   // one request's fields together: 64 bytes
  std::int64_t id;
  std::int32_t prompt_len, output_len;
  float temperature, top_p;
  std::int32_t num_blocks, priority;
  bool finished, streaming;
  char padding[30];
};

struct RequestsSoA {  // one array per field
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
  // the scheduler cares about one field each step: counting the tokens every request has generated
  double t_aos = best_ms([&] { std::int64_t s = 0; for (const auto& r : aos) s += r.output_len; sink = s; });
  double t_soa = best_ms([&] { std::int64_t s = 0; for (int v : soa.output_len) s += v; sink = s; });
  std::printf("sizeof(RequestAoS)=%zu\n", sizeof(RequestAoS));
  std::printf("只读一个字段：AoS %.2f ms，SoA %.2f ms\n", t_aos, t_soa);
}
```

```text title="one run (an x86 server)"
sizeof(RequestAoS)=64
只读一个字段：AoS 20.58 ms，SoA 2.17 ms
```

About 10 times apart, which matches the ratio of "useful bytes per cache line" (64 / 4 = 16, and SoA vectorizes as well). Conversely, when **all** of a request's fields are read every time, AoS is better: one cache line brings everything.
An inference engine's "batch metadata" (each request's sequence length, block table and sampling parameters) is mostly organized as SoA, because it ends up as the contiguous arrays passed to GPU kernels (`seq_lens`, `block_tables`, `temperatures`).

### The order of access {#访问顺序}

The same data accessed in a different order performs entirely differently. Traversing a row-major matrix by column crosses a whole row on each access and uses 4 bytes of each cache line:

```cpp title="traverse.cpp" sanitize="none" flags="-O2"
// g++ -std=c++20 -O2 traverse.cpp && ./a.out
#include <chrono>
#include <cstdio>
#include <vector>

volatile float sink;

int main() {
  const int n = 4096;
  std::vector<float> m(std::size_t(n) * n, 1.0f);   // row-major: m[i * n + j]
  auto time = [&](bool by_row) {
    auto t0 = std::chrono::steady_clock::now();
    float s = 0;
    for (int a = 0; a < n; ++a)
      for (int b = 0; b < n; ++b) s += by_row ? m[std::size_t(a) * n + b] : m[std::size_t(b) * n + a];
    sink = s;
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  time(true);   // warm-up
  std::printf("按行遍历 %.0f ms，按列遍历 %.0f ms\n", time(true), time(false));
}
```

```text title="one run (an x86 server)"
按行遍历 23 ms，按列遍历 175 ms
```

A KV cache's layout is this question: `[token][head][dim]` or `[head][token][dim]`, depending on which dimension the kernel reads along most often. Paged KV putting one block's data (16 tokens, say) together is exactly so that reading KV by block during decode stays a contiguous, large access.

## False sharing {#伪共享}

Cores keep their cache lines consistent through a coherence protocol (MESI): once one core writes a cache line, the copies in the other cores are invalidated. If two threads write **different variables** that happen to sit in **the same cache line**, that line is invalidated and shuttled back and forth between the cores, which is **false sharing**:

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
struct alignas(64) Padded {   // each counter gets a 64-byte cache line to itself
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

```text title="one run (an x86 server)"
sizeof(Plain)=8 sizeof(Padded)=64
4 个线程各加自己的计数器 20000000 次：挨在一起 1864 ms，按缓存行隔开 137 ms
```

The four threads share nothing logically, and it is an order of magnitude slower purely because the counters sit next to each other. Per-thread statistics, a lock-free queue's read and write pointers, and each worker's state in a thread pool should all be separated by a cache line.
C++17 offers `std::hardware_destructive_interference_size` for that distance, but its value depends on the compile options (g++ warns when it is used), and in practice 64 is usually written directly (128 on some ARM servers).

!!! interview "Answering in an interview"
    On layout and the cache: a struct's size includes alignment padding, so `{bool; double; int}` is 24 bytes and `{double; int; bool}`, ordered by alignment, is 16; a layout shared across languages or devices is nailed down with `static_assert` on the size and the offsets. Pinned memory, RDMA registration and `O_DIRECT` need page alignment. Performance comes down to how many bytes of each cache line are useful: reading a few fields, a struct of arrays (SoA) beats an array of structs (AoS) by an order of magnitude, and the traversal order has to match the storage order, which is the same thing as a GPU's coalesced access. Variables written often by different threads are separated by a cache line (64 bytes) to avoid false sharing.

## Exercises {#练习}

1. Below is a request's state in the scheduler. Reorder the fields to make it as small as possible, give the `sizeof` after reordering, and pin it down with `static_assert`.

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

??? success "Answer"
    The original layout: 7 bytes of padding after `is_prefill`, 2 after `lora_slot`, 3 after `finished` and 7 after `priority`, for 48 bytes in all. Ordered by alignment, largest first: two 8-byte fields, two 4-byte, one 2-byte and three 1-byte, which is exactly $16 + 8 + 2 + 3 = 29$, rounded up to 32:

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

    ```text title="output"
    重排前 48 字节，重排后 32 字节
    ```

2. One layer's KV cache is stored as bf16 in `[num_blocks][block_size=16][num_kv_heads=8][head_dim=128]`. A decode kernel reads all 16 tokens' K of one block for **one KV head**: how many contiguous bytes does it get per token? How many 64-byte cache lines does it read in all (counting CPU cache lines)? And with the layout changed to `[num_blocks][num_kv_heads][block_size][head_dim]`?

??? success "Answer"
    The original layout: one token's one head is $128 \times 2 = 256$ contiguous bytes, exactly 4 cache lines, and adjacent tokens are $8 \times 256 = 2048$ bytes apart. The 16 tokens are $16 \times 4 = 64$ cache lines in all, but in 16 separate runs.
    With the head outside the block, one head's 16 tokens form a single run of $16 \times 256 = 4096$ bytes, still 64 cache lines but one contiguous access, which suits hardware prefetching and a GPU's large loads (the TMA, `cp.async`) far better.
    Whether vLLM's paged KV uses `[num_blocks, block_size, num_kv_heads, head_size]` or `[num_blocks, num_kv_heads, block_size, head_size]` is chosen on exactly this reasoning in its different attention back ends (see the Inference Systems handbook's [paged KV cache](serving://engine/paged-kv/)).

## Summary {#小结}

- [x] A struct's size includes alignment padding; ordering the fields by alignment, largest first, removes most of it; a layout shared across languages or devices is nailed down with `static_assert`.
- [x] Vectorized access needs 16-byte alignment, and pinning, RDMA and `O_DIRECT` need page alignment; use `alignas`, `aligned_alloc` or an aligned allocator.
- [x] Performance comes down to how many bytes of each cache line are useful: reading a few fields, SoA beats AoS by an order of magnitude, and the traversal order has to match the storage order. This is the same thing as a GPU's coalesced access.
- [x] Variables written often by different threads are separated by a cache line (64 bytes) to avoid false sharing.
