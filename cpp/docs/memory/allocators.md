# 分配器与内存池

<p class="lead">推理系统几乎从不在热路径上调用 <code>malloc</code> 或 <code>cudaMalloc</code>：KV Cache 在启动时一次性分配好、切成块自己管理；PyTorch 把释放的显存留着复用；每一步的临时缓冲区从一块预先分配的内存里线性切出来。这一章用 C++ 实现这几种内存管理方式，它们正是推理引擎里内存管理的核心。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么不能在每一步 decode 时 `cudaMalloc` / `cudaFree` 临时缓冲区？
    2. arena（线性分配器）怎么释放单个对象？
    3. 分页 KV 的块分配器为什么用"空闲列表 + 引用计数"？写时复制在什么时候发生？
    4. PyTorch 显示"reserved 70 GB、allocated 50 GB"，却在申请 4 GB 时报 OOM，可能是什么原因？
    5. 怎样让 `std::vector` 从你指定的内存里分配？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `cudaMalloc` / `cudaFree` 很慢，而且会引起设备同步，打断 CPU 和 GPU 的流水，还会产生碎片；decode 每一步都调用，开销和抖动都不可接受。应该事先分配好，或者用内存池复用。
    2. 不释放单个对象：分配只是把偏移量往后挪，释放单个对象什么都不做；用完之后 `reset()` 一次性回收全部。所以只放不需要析构的对象。
    3. 块大小固定，空闲列表（栈）让分配和释放都是 O(1)，也没有外部碎片；引用计数让多个请求共享同一个块（前缀缓存、并行采样）。写时复制发生在某个请求要往一个被共享（引用计数大于 1）的块里写入的时候：先分配新块、复制内容，再写。
    4. 碎片：reserved 里空闲的 20 GB 分散成很多小块，凑不出一块连续的 4 GB；缓存分配器按大小缓存的块不能跨段合并。可以看内存快照，开 `expandable_segments`，或者减少大小不一的临时分配。
    5. 用 `std::pmr::vector<T>`，构造时传一个 `std::pmr::memory_resource`（比如 `monotonic_buffer_resource` 指向你的缓冲区）；或者写一个自定义的分配器类型作为 `vector` 的第二个模板参数。

## 为什么要自己管理内存

通用分配器要应付任意大小、任意顺序的申请和释放，代价是：

- **慢**：`malloc` 可能要加锁、查找合适的空闲块；`cudaMalloc` 更慢，而且会**隐式同步整个设备**，在 decode 的每一步里调用一次就足以让 GPU 频繁空转；
- **碎片**：大小不一的块反复申请释放，总空闲内存够，却找不到一块连续的；
- **不可预测**：一次偶发的慢分配就是一次尾延迟的毛刺。

而推理系统的内存使用模式恰好很有规律：KV Cache 按固定大小的块使用，每一步的临时缓冲区用完就扔，激活的大小只取决于 batch 的形状。针对规律设计的分配器，可以做到 O(1)、零碎片、零系统调用。

## arena：用完一起扔

**arena**（也叫线性分配器、bump allocator）持有一大块内存和一个偏移量：分配就是把偏移量往后挪（按对齐要求向上取整），释放单个对象什么都不做，用完之后 `reset()` 一次性全部回收。适合"每一步的临时数据"：

```cpp title="arena.cpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>

class Arena {
 public:
  explicit Arena(std::size_t capacity)   // capacity 必须是 4096 的倍数
      : buf_(static_cast<std::byte*>(std::aligned_alloc(4096, capacity))), cap_(capacity) {
    if (!buf_) throw std::bad_alloc();
  }
  void* allocate(std::size_t n, std::size_t align = alignof(std::max_align_t)) {
    auto base = reinterpret_cast<std::uintptr_t>(buf_.get());
    std::uintptr_t p = (base + off_ + align - 1) & ~(std::uintptr_t(align) - 1);   // 向上对齐（align 是 2 的幂）
    std::size_t end = p - base + n;
    if (end > cap_) throw std::bad_alloc();
    off_ = end;
    peak_ = std::max(peak_, off_);
    return reinterpret_cast<void*>(p);
  }
  template <class T>
  T* alloc_array(std::size_t n) { return static_cast<T*>(allocate(n * sizeof(T), alignof(T))); }
  void reset() { off_ = 0; }   // 一次性"释放"全部
  std::size_t used() const { return off_; }
  std::size_t peak() const { return peak_; }

 private:
  struct Free {
    void operator()(std::byte* p) const { std::free(p); }
  };
  std::unique_ptr<std::byte, Free> buf_;
  std::size_t cap_, off_ = 0, peak_ = 0;
};

int main() {
  Arena arena(1 << 20);
  const void* first = nullptr;
  for (int step = 0; step < 3; ++step) {
    int batch = 8 + step * 8;   // 每一步的 batch 大小不同
    auto* positions = arena.alloc_array<std::int32_t>(batch);
    auto* slot_mapping = arena.alloc_array<std::int64_t>(batch);
    auto* logits = static_cast<float*>(arena.allocate(batch * 1024 * sizeof(float), 64));   // 按缓存行对齐
    for (int i = 0; i < batch; ++i) {
      positions[i] = i;
      slot_mapping[i] = i;
      logits[i] = 0;
    }
    if (step == 0) first = positions;
    std::printf("step %d：batch=%d 用了 %zu 字节，positions 与第一步同地址：%s\n", step, batch, arena.used(),
                positions == first ? "是" : "否");
    arena.reset();
  }
  std::printf("峰值 %zu 字节\n", arena.peak());
}
```

```text title="输出"
step 0：batch=8 用了 32896 字节，positions 与第一步同地址：是
step 1：batch=16 用了 65728 字节，positions 与第一步同地址：是
step 2：batch=24 用了 98624 字节，positions 与第一步同地址：是
峰值 98624 字节
```

三步用的是同一块内存，没有任何系统调用。注意 `logits` 按 64 字节对齐，所以在 `slot_mapping` 后面留了空隙（第一步从 96 对齐到 128）。

arena 只能存放**不需要析构**的对象（`int`、`float`、POD 结构体），或者你自己负责在 `reset()` 之前调用析构函数。推理引擎里传给 kernel 的那些元数据数组（位置、槽位映射、序列长度）正好都是这种。

## 块分配器：分页 KV 的核心

KV Cache 在启动时按"显存预算 ÷ 每块字节数"一次性分配好，切成固定大小的块（比如每块 16 个 token）。之后的分配、释放只是在**块号**上操作：

- **空闲列表**：一个装着空闲块号的栈，分配是 `pop`，释放是 `push`，都是 O(1)，块大小固定所以没有碎片；
- **引用计数**：多个序列可以共享同一个块——共享前缀、并行采样（`n > 1`）、beam search 分叉。计数降到 0 才真正回到空闲列表；
- **写时复制**：一个序列要往一个**被共享**的块里写新 token 时，先分配一个新块、复制内容、再写，避免改掉别人的数据。

```cpp title="block_pool.cpp"
#include <cstdio>
#include <vector>

class BlockPool {
 public:
  explicit BlockPool(int n) : ref_(n, 0) {
    for (int i = n - 1; i >= 0; --i) free_.push_back(i);   // 栈顶是 0：小编号先分配
  }
  int allocate() {
    if (free_.empty()) return -1;   // 真实系统在这里触发抢占或排队
    int b = free_.back();
    free_.pop_back();
    ref_[b] = 1;
    return b;
  }
  void retain(int b) { ++ref_[b]; }   // 又一个序列共享这个块
  void release(int b) {
    if (--ref_[b] == 0) free_.push_back(b);
  }
  // 写之前调用：块被共享时复制一份给自己（写时复制），返回可以写的块号
  int make_writable(int b) {
    if (ref_[b] == 1) return b;
    int nb = allocate();
    if (nb < 0) return -1;
    // copy_kv(b, nb);   真实系统在这里用一个小 kernel 复制块的 KV 数据
    release(b);
    return nb;
  }
  int num_free() const { return int(free_.size()); }
  int refcount(int b) const { return ref_[b]; }

 private:
  std::vector<int> free_;
  std::vector<int> ref_;
};

int main() {
  BlockPool pool(8);
  std::vector<int> a = {pool.allocate(), pool.allocate(), pool.allocate()};   // 请求 A：3 个块
  std::printf("A=[%d %d %d] 剩余 %d\n", a[0], a[1], a[2], pool.num_free());

  std::vector<int> b = a;   // B 从 A 分叉（并行采样 / beam search）：共享全部块
  for (int blk : b) pool.retain(blk);
  std::printf("分叉后块 2 的引用计数=%d 剩余 %d\n", pool.refcount(2), pool.num_free());

  b.back() = pool.make_writable(b.back());   // B 要往最后一个块里写新 token
  std::printf("B 写之前复制：B=[%d %d %d] 块 2 引用计数=%d 剩余 %d\n", b[0], b[1], b[2], pool.refcount(2),
              pool.num_free());

  for (int blk : a) pool.release(blk);   // A 结束
  std::printf("A 结束：块 0 引用计数=%d 剩余 %d\n", pool.refcount(0), pool.num_free());
  for (int blk : b) pool.release(blk);
  std::printf("B 结束：剩余 %d\n", pool.num_free());
}
```

```text title="输出"
A=[0 1 2] 剩余 5
分叉后块 2 的引用计数=2 剩余 5
B 写之前复制：B=[0 1 3] 块 2 引用计数=1 剩余 4
A 结束：块 0 引用计数=1 剩余 5
B 结束：剩余 8
```

只有最后一个（正在写的）块需要复制，前面写满的块一直共享。vLLM 的 `BlockPool`、SGLang 的 `TokenToKVPool` 都是这个结构，只是块号换成了显存里的偏移。
推理系统手册的[分页 KV Cache](serving://engine/paged-kv/) 和 mini-sglang 的 [KV 池与 page table](minisgl://compute/kvcache/) 用 Python 实现了同样的逻辑；练习题里的[块分配器](root://practice/#/p/sv-block-pool-cow)可以对照着做。

注意这里的引用计数是普通的 `int`，不是 `shared_ptr` 那样的原子计数：块分配器只在调度器线程里被访问，不需要原子操作——这也是推理引擎不用 `shared_ptr` 管 KV 块的原因之一。

## 缓存分配器：释放了也不还

PyTorch 在 GPU 上用的是**缓存分配器**：`free` 之后不调用 `cudaFree`，而是把这块显存按大小放进空闲池，下次申请差不多大小的内存时直接复用。
为了提高复用率，申请的大小会先向上取整：小块取整到 512 字节的倍数，大块取整到 2 MB 的倍数。下面是一个简化版（用 `malloc` 代替 `cudaMalloc`）：

```cpp title="caching_allocator.cpp"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <unordered_map>
#include <vector>

class CachingAllocator {
 public:
  void* allocate(std::size_t n) {
    std::size_t sz = round_up(n);
    auto& bin = free_[sz];
    void* p;
    if (!bin.empty()) {
      p = bin.back();
      bin.pop_back();
      ++hits_;
    } else {
      p = std::malloc(sz);   // 真实系统里这里是 cudaMalloc：慢，而且会同步
      if (!p) throw std::bad_alloc();
      ++raw_allocs_;
      reserved_ += sz;
    }
    sizes_[p] = sz;
    return p;
  }
  void deallocate(void* p) {   // 不还给系统，按大小放回空闲池
    std::size_t sz = sizes_.at(p);
    sizes_.erase(p);
    free_[sz].push_back(p);
  }
  void empty_cache() {         // 对应 torch.cuda.empty_cache()
    for (auto& [sz, bin] : free_) {
      for (void* p : bin) {
        std::free(p);
        reserved_ -= sz;
      }
      bin.clear();
    }
  }
  ~CachingAllocator() { empty_cache(); }

  static std::size_t round_up(std::size_t n) {
    const std::size_t k = n < (1 << 20) ? 512 : (2 << 20);
    return (n + k - 1) / k * k;
  }
  int raw_allocs() const { return raw_allocs_; }
  int hits() const { return hits_; }
  double reserved_mib() const { return reserved_ / 1048576.0; }

 private:
  std::unordered_map<std::size_t, std::vector<void*>> free_;
  std::unordered_map<void*, std::size_t> sizes_;
  std::size_t reserved_ = 0;
  int raw_allocs_ = 0, hits_ = 0;
};

int main() {
  CachingAllocator alloc;
  const int batches[] = {32, 48, 32, 48};
  for (int step = 0; step < 4; ++step) {
    std::size_t b = batches[step];
    void* hidden = alloc.allocate(b * 4096 * 2);    // [batch, 4096] 的 bf16 激活
    void* logits = alloc.allocate(b * 32000 * 4);   // [batch, 32000] 的 fp32 logits
    void* meta = alloc.allocate(b * 8);             // 每个请求一个 int64
    alloc.deallocate(meta);
    alloc.deallocate(logits);
    alloc.deallocate(hidden);
    std::printf("step %d：batch=%zu，底层分配累计 %d 次，复用累计 %d 次，reserved %.2f MiB\n", step, b,
                alloc.raw_allocs(), alloc.hits(), alloc.reserved_mib());
  }
  alloc.empty_cache();
  std::printf("empty_cache 之后 reserved %.2f MiB\n", alloc.reserved_mib());
}
```

```text title="输出"
step 0：batch=32，底层分配累计 3 次，复用累计 0 次，reserved 4.25 MiB
step 1：batch=48，底层分配累计 5 次，复用累计 1 次，reserved 10.63 MiB
step 2：batch=32，底层分配累计 5 次，复用累计 4 次，reserved 10.63 MiB
step 3：batch=48，底层分配累计 5 次，复用累计 7 次，reserved 10.63 MiB
empty_cache 之后 reserved 0.00 MiB
```

batch 大小一旦出现过，之后的每一步都不再有底层分配。代价是：`nvidia-smi` 看到的占用（reserved）比实际在用的（allocated）大；而且按大小分桶的空闲块**不能拼起来用**——
空闲池里有 20 个 2 MB 的块，也满足不了一次 40 MB 的申请，这就是"明明还有空闲显存却 OOM"的常见原因（真实的 PyTorch 分配器还会拆分大块、合并相邻的空闲块，但碎片问题依然存在）。

推理引擎的做法是在启动时就把 KV Cache 按显存预算一次性分配走（vLLM 的 `gpu_memory_utilization`），并用 CUDA Graph 固定住 decode 阶段的所有临时缓冲区，运行时几乎不再触发分配器。

## 让标准容器用你的内存：`std::pmr`

C++17 的 `std::pmr`（多态内存资源）让标准容器从指定的"内存资源"里分配，而不用改容器的类型参数：`std::pmr::vector<int>` 可以从 arena、内存池或者栈上的缓冲区分配。
下面重载全局的 `operator new` 来统计堆分配的次数，对比普通的 `vector` 和从栈上缓冲区分配的 `pmr::vector`：

```cpp title="pmr.cpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory_resource>
#include <new>
#include <vector>

static int g_allocs = 0;
void* operator new(std::size_t n) {
  ++g_allocs;
  if (void* p = std::malloc(n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

std::size_t step_std() {
  std::vector<int> ids;
  for (int i = 0; i < 1000; ++i) ids.push_back(i);
  std::vector<float> w(512);
  return ids.size() + w.size();
}

std::size_t step_pmr(std::pmr::memory_resource* mr) {
  std::pmr::vector<int> ids(mr);
  for (int i = 0; i < 1000; ++i) ids.push_back(i);
  std::pmr::vector<float> w(512, mr);
  return ids.size() + w.size();
}

int main() {
  int before = g_allocs;
  for (int s = 0; s < 3; ++s) step_std();
  std::printf("std::vector：3 步共 %d 次堆分配\n", g_allocs - before);

  alignas(64) static std::byte buf[64 * 1024];
  before = g_allocs;
  for (int s = 0; s < 3; ++s) {
    // 每一步在同一块缓冲区上建一个单调资源：只分配不释放，析构时整体丢弃
    std::pmr::monotonic_buffer_resource step_mem(buf, sizeof buf, std::pmr::null_memory_resource());
    step_pmr(&step_mem);
  }
  std::printf("pmr::vector + 预分配缓冲区：3 步共 %d 次堆分配\n", g_allocs - before);
}
```

```text title="输出"
std::vector：3 步共 36 次堆分配
pmr::vector + 预分配缓冲区：3 步共 0 次堆分配
```

每步 12 次 = `ids` 从容量 1 增长到 1024 的 11 次 + `w` 的 1 次。`monotonic_buffer_resource` 就是标准库版本的 arena；缓冲区用完时它会向"上游资源"要内存，这里上游是 `null_memory_resource()`（直接抛异常），保证绝不会悄悄地退回到堆分配。
标准库还提供了 `unsynchronized_pool_resource` / `synchronized_pool_resource`，是按大小分桶的内存池。

## 练习

1. 写一个对齐分配器 `AlignedAllocator<T, Align>`，让 `std::vector<float, AlignedAllocator<float, 64>>` 的缓冲区按 64 字节对齐，`AlignedAllocator<float, 4096>` 按页对齐。

??? success "参考答案"
    分配器只需要 `value_type`、`allocate`、`deallocate` 和相等比较；因为模板有一个非类型参数，还要提供 `rebind`，告诉容器怎样得到"同样对齐方式、别的元素类型"的分配器。对齐的分配用 C++17 的 `operator new(size, std::align_val_t)`：

    ```cpp title="aligned_allocator.cpp"
    #include <cstddef>
    #include <cstdint>
    #include <cstdio>
    #include <new>
    #include <vector>

    template <class T, std::size_t Align>
    struct AlignedAllocator {
      using value_type = T;
      template <class U>
      struct rebind {
        using other = AlignedAllocator<U, Align>;
      };
      AlignedAllocator() = default;
      template <class U>
      AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

      T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Align})); }
      void deallocate(T* p, std::size_t n) noexcept { ::operator delete(p, n * sizeof(T), std::align_val_t{Align}); }
      template <class U>
      bool operator==(const AlignedAllocator<U, Align>&) const noexcept { return true; }
    };

    template <class V>
    bool aligned(const V& v, std::size_t a) { return reinterpret_cast<std::uintptr_t>(v.data()) % a == 0; }

    int main() {
      std::vector<float, AlignedAllocator<float, 64>> v(1000);
      std::vector<float, AlignedAllocator<float, 4096>> page(1024);
      v.resize(5000);   // 扩容后仍然对齐
      std::printf("64 字节对齐：%s，页对齐：%s\n", aligned(v, 64) ? "是" : "否", aligned(page, 4096) ? "是" : "否");
    }
    ```

    ```text title="输出"
    64 字节对齐：是，页对齐：是
    ```

2. 线上的推理服务运行一段时间后，`torch.cuda.memory_reserved()` 是 70 GB，`memory_allocated()` 是 50 GB，这时申请一个 4 GB 的张量报了 OOM。解释原因，并给出至少两个缓解办法。

??? success "参考答案"
    reserved 和 allocated 之差（20 GB）是缓存分配器留着复用的空闲块，但它们是很多个大小不一的块，没有一块（或相邻可合并的一段）能满足 4 GB 的连续申请——这是外部碎片。
    缓解办法：
    让形状稳定下来，减少不同大小的申请（padding 到固定的几个 batch 大小、用 CUDA Graph 固定临时缓冲区）；
    在启动时一次性预分配大块（KV Cache、工作区），运行时从自己的池里切；
    开启 PyTorch 的 `expandable_segments`（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`），它用虚拟内存映射让段可以增长，减少碎片；
    在空闲的时机调用 `torch.cuda.empty_cache()` 把空闲块还给驱动（代价是之后要重新 `cudaMalloc`）。

## 小结

- [x] 热路径上不要调用通用分配器，尤其是 `cudaMalloc`：慢、会同步、产生碎片。
- [x] arena：分配就是挪偏移，一次性回收，适合每一步的临时数据；只放不需要析构的对象。
- [x] 固定大小的块分配器：空闲栈 O(1) 分配释放、没有碎片；引用计数支持共享，写之前对共享块做写时复制——这就是分页 KV 的内存管理。
- [x] 缓存分配器把释放的块按大小留着复用，代价是 reserved 高于 allocated，以及碎片导致的 OOM。
- [x] `std::pmr` 让标准容器从指定的内存资源分配；自定义分配器要提供 `allocate`、`deallocate`、相等比较，模板有非类型参数时还要 `rebind`。
