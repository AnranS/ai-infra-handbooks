# Allocators and memory pools

<p class="lead">An inference system almost never calls <code>malloc</code> or <code>cudaMalloc</code> on the hot path: the KV cache is allocated once at startup, cut into blocks and managed by hand; PyTorch keeps the device memory it frees for reuse; each step's temporary buffers are carved linearly out of one preallocated block. This chapter implements these ways of managing memory in C++, and they are the heart of an inference engine's memory management.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why can a temporary buffer not be `cudaMalloc`ed and `cudaFree`d at each decode step?
    2. How does an arena (a linear allocator) free a single object?
    3. Why does paged KV's block allocator use "a free list plus reference counts"? When does copy-on-write happen?
    4. PyTorch shows "70 GB reserved, 50 GB allocated" and still reports OOM on a 4 GB request. What could be the reason?
    5. How do you make a `std::vector` allocate from memory you choose?

??? success "Answers (try it yourself first, then expand)"
    1. `cudaMalloc` / `cudaFree` are slow, synchronize the device, interrupt the CPU-GPU pipeline and fragment memory; called at every decode step, both the cost and the jitter are unacceptable. Allocate beforehand, or reuse through a memory pool.
    2. It does not free a single object: allocation only moves the offset forward and freeing one object does nothing; a single `reset()` reclaims everything afterwards. So only objects that need no destructor go in it.
    3. The block size is fixed, so a free list (a stack) makes allocation and freeing O(1) with no external fragmentation; reference counts let several requests share a block (prefix caching, parallel sampling). Copy-on-write happens when a request is about to write into a block that is shared (its reference count above 1): a new block is allocated, the contents copied, and then written.
    4. Fragmentation: the 20 GB of free space inside reserved is scattered into many small blocks that cannot add up to one contiguous 4 GB; the caching allocator's size-bucketed blocks cannot be merged across segments. Read the memory snapshot, turn on `expandable_segments`, or make fewer temporary allocations of varying size.
    5. Use `std::pmr::vector<T>` and pass a `std::pmr::memory_resource` at construction (a `monotonic_buffer_resource` over your buffer, say); or write a custom allocator type as the `vector`'s second template argument.

## Why manage memory yourself {#为什么要自己管理内存}

A general-purpose allocator has to cope with requests and releases of any size in any order, at the cost of:

- **being slow**: `malloc` may take a lock and search for a suitable free block; `cudaMalloc` is slower still and **implicitly synchronizes the whole device**, and calling it once per decode step is enough to leave the GPU idling often;
- **fragmentation**: blocks of varying size requested and released repeatedly leave enough free memory in total but nothing contiguous;
- **unpredictability**: one occasional slow allocation is one spike in the tail latency.

An inference system's memory use, meanwhile, is very regular: the KV cache is used in fixed-size blocks, each step's temporary buffers are thrown away when the step ends, and the activations' size depends only on the batch's shape. An allocator designed for that regularity can be O(1), fragmentation-free and free of system calls.

## An arena: thrown away all at once {#arena用完一起扔}

![Figure: an arena - allocation is pushing a pointer forward, and the whole block is thrown away at the end](../assets/figures/arena-bump.svg){.aig-svg}

An **arena** (also a linear or bump allocator) holds one large block of memory and an offset: allocation moves the offset forward (rounded up to the alignment), freeing one object does nothing, and a `reset()` reclaims everything at the end. It suits "each step's temporary data":

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
  explicit Arena(std::size_t capacity)   // capacity has to be a multiple of 4096
      : buf_(static_cast<std::byte*>(std::aligned_alloc(4096, capacity))), cap_(capacity) {
    if (!buf_) throw std::bad_alloc();
  }
  void* allocate(std::size_t n, std::size_t align = alignof(std::max_align_t)) {
    auto base = reinterpret_cast<std::uintptr_t>(buf_.get());
    std::uintptr_t p = (base + off_ + align - 1) & ~(std::uintptr_t(align) - 1);   // round up (align is a power of two)
    std::size_t end = p - base + n;
    if (end > cap_) throw std::bad_alloc();
    off_ = end;
    peak_ = std::max(peak_, off_);
    return reinterpret_cast<void*>(p);
  }
  template <class T>
  T* alloc_array(std::size_t n) { return static_cast<T*>(allocate(n * sizeof(T), alignof(T))); }
  void reset() { off_ = 0; }   // "free" everything at once
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
    int batch = 8 + step * 8;   // a different batch size each step
    auto* positions = arena.alloc_array<std::int32_t>(batch);
    auto* slot_mapping = arena.alloc_array<std::int64_t>(batch);
    auto* logits = static_cast<float*>(arena.allocate(batch * 1024 * sizeof(float), 64));   // aligned to a cache line
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

```text title="output"
step 0：batch=8 用了 32896 字节，positions 与第一步同地址：是
step 1：batch=16 用了 65728 字节，positions 与第一步同地址：是
step 2：batch=24 用了 98624 字节，positions 与第一步同地址：是
峰值 98624 字节
```

The three steps use the same memory with no system call at all. Note that `logits` is aligned to 64 bytes, leaving a gap after `slot_mapping` (the first step rounds 96 up to 128).

An arena can only hold objects that **need no destructor** (`int`, `float`, POD structs), or you call the destructors yourself before the `reset()`. The metadata arrays an inference engine passes to its kernels (positions, the slot mapping, sequence lengths) are exactly of that kind.

## A block allocator: the heart of paged KV {#块分配器分页-kv-的核心}

![Figure: a paged KV cache - fixed-size blocks, the block table and the free list](../assets/figures/paged-kv.svg){.aig-svg}

The KV cache is allocated once at startup as "the memory budget ÷ the bytes per block" and cut into fixed-size blocks (16 tokens each, say). Allocation and freeing afterwards only work on **block numbers**:

- **the free list**: a stack of free block numbers, where allocation is a `pop` and freeing a `push`, both O(1), with a fixed block size so there is no fragmentation;
- **reference counts**: several sequences can share a block, through a shared prefix, parallel sampling (`n > 1`) or a beam search fork. Only when the count reaches 0 does the block really return to the free list;
- **copy-on-write**: when a sequence is about to write a new token into a **shared** block, it allocates a new block, copies the contents and then writes, so as not to change somebody else's data.

```cpp title="block_pool.cpp"
#include <cstdio>
#include <vector>

class BlockPool {
 public:
  explicit BlockPool(int n) : ref_(n, 0) {
    for (int i = n - 1; i >= 0; --i) free_.push_back(i);   // 0 is on top of the stack: the low numbers go out first
  }
  int allocate() {
    if (free_.empty()) return -1;   // a real system preempts or queues here
    int b = free_.back();
    free_.pop_back();
    ref_[b] = 1;
    return b;
  }
  void retain(int b) { ++ref_[b]; }   // another sequence shares this block
  void release(int b) {
    if (--ref_[b] == 0) free_.push_back(b);
  }
  // called before writing: when the block is shared, copy one for ourselves (copy-on-write) and return a block number we may write
  int make_writable(int b) {
    if (ref_[b] == 1) return b;
    int nb = allocate();
    if (nb < 0) return -1;
    // copy_kv(b, nb);   a real system copies the block's KV data with a small kernel here
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
  std::vector<int> a = {pool.allocate(), pool.allocate(), pool.allocate()};   // request A: 3 blocks
  std::printf("A=[%d %d %d] 剩余 %d\n", a[0], a[1], a[2], pool.num_free());

  std::vector<int> b = a;   // B forks from A (parallel sampling / beam search): every block shared
  for (int blk : b) pool.retain(blk);
  std::printf("分叉后块 2 的引用计数=%d 剩余 %d\n", pool.refcount(2), pool.num_free());

  b.back() = pool.make_writable(b.back());   // B is about to write a new token into the last block
  std::printf("B 写之前复制：B=[%d %d %d] 块 2 引用计数=%d 剩余 %d\n", b[0], b[1], b[2], pool.refcount(2),
              pool.num_free());

  for (int blk : a) pool.release(blk);   // A finishes
  std::printf("A 结束：块 0 引用计数=%d 剩余 %d\n", pool.refcount(0), pool.num_free());
  for (int blk : b) pool.release(blk);
  std::printf("B 结束：剩余 %d\n", pool.num_free());
}
```

```text title="output"
A=[0 1 2] 剩余 5
分叉后块 2 的引用计数=2 剩余 5
B 写之前复制：B=[0 1 3] 块 2 引用计数=1 剩余 4
A 结束：块 0 引用计数=1 剩余 5
B 结束：剩余 8
```

Only the last block (the one being written) has to be copied, and the full blocks before it stay shared. vLLM's `BlockPool` and SGLang's `TokenToKVPool` have this structure, with the block number replaced by an offset into device memory.
The Inference Systems handbook's [paged KV cache](serving://engine/paged-kv/) and mini-sglang's [the KV pool and the page table](minisgl://compute/kvcache/) implement the same logic in Python; the exercise on [a block allocator](root://practice/#/p/sv-block-pool-cow) is worth doing alongside.

Note that the reference count here is an ordinary `int` and not an atomic one like a `shared_ptr`'s: the block allocator is touched only by the scheduler thread and needs no atomics, which is one reason inference engines do not manage KV blocks with `shared_ptr`.

## A caching allocator: freed but not returned {#缓存分配器释放了也不还}

What PyTorch uses on a GPU is a **caching allocator**: a `free` does not call `cudaFree` but puts that memory into a free pool by size, to be reused directly the next time a request of about that size comes.
To improve the reuse, a request's size is rounded up first: small blocks to a multiple of 512 bytes and large ones to a multiple of 2 MB. Below is a simplified version (with `malloc` standing in for `cudaMalloc`):

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
      p = std::malloc(sz);   // a real system has cudaMalloc here: slow, and it synchronizes
      if (!p) throw std::bad_alloc();
      ++raw_allocs_;
      reserved_ += sz;
    }
    sizes_[p] = sz;
    return p;
  }
  void deallocate(void* p) {   // not returned to the system, but put back in the free pool by size
    std::size_t sz = sizes_.at(p);
    sizes_.erase(p);
    free_[sz].push_back(p);
  }
  void empty_cache() {         // the counterpart of torch.cuda.empty_cache()
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
    void* hidden = alloc.allocate(b * 4096 * 2);    // [batch, 4096] of bf16 activations
    void* logits = alloc.allocate(b * 32000 * 4);   // [batch, 32000] of fp32 logits
    void* meta = alloc.allocate(b * 8);             // one int64 per request
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

```text title="output"
step 0：batch=32，底层分配累计 3 次，复用累计 0 次，reserved 4.25 MiB
step 1：batch=48，底层分配累计 5 次，复用累计 1 次，reserved 10.63 MiB
step 2：batch=32，底层分配累计 5 次，复用累计 4 次，reserved 10.63 MiB
step 3：batch=48，底层分配累计 5 次，复用累计 7 次，reserved 10.63 MiB
empty_cache 之后 reserved 0.00 MiB
```

Once a batch size has been seen, no step after it reaches the underlying allocator. The costs: what `nvidia-smi` shows (reserved) is larger than what is in use (allocated); and free blocks bucketed by size **cannot be put together**.
Twenty 2 MB blocks in the free pool still cannot satisfy one 40 MB request, which is the common reason for "OOM while device memory is clearly free" (the real PyTorch allocator also splits large blocks and merges adjacent free ones, but the fragmentation problem remains).

What an inference engine does is allocate the whole KV cache at startup by the memory budget (vLLM's `gpu_memory_utilization`) and pin down every temporary buffer of the decode phase with a CUDA Graph, so the allocator is hardly reached at run time.

## Letting the standard containers use your memory: `std::pmr` {#让标准容器用你的内存stdpmr}

C++17's `std::pmr` (polymorphic memory resources) lets a standard container allocate from a chosen "memory resource" without changing the container's type arguments: a `std::pmr::vector<int>` can allocate from an arena, a memory pool or a buffer on the stack.
Below, the global `operator new` is overloaded to count heap allocations, comparing an ordinary `vector` with a `pmr::vector` allocating from a stack buffer:

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
    // a monotonic resource over the same buffer each step: it only allocates, never frees, and discards everything on destruction
    std::pmr::monotonic_buffer_resource step_mem(buf, sizeof buf, std::pmr::null_memory_resource());
    step_pmr(&step_mem);
  }
  std::printf("pmr::vector + 预分配缓冲区：3 步共 %d 次堆分配\n", g_allocs - before);
}
```

```text title="output"
std::vector：3 步共 36 次堆分配
pmr::vector + 预分配缓冲区：3 步共 0 次堆分配
```

The 12 per step are `ids` growing from capacity 1 to 1024 in 11 steps plus 1 for `w`. A `monotonic_buffer_resource` is the standard library's arena; when its buffer runs out it asks an "upstream resource" for more, and here the upstream is `null_memory_resource()` (which throws outright), guaranteeing it never quietly falls back to the heap.
The standard library also provides `unsynchronized_pool_resource` / `synchronized_pool_resource`, which are memory pools bucketed by size.

!!! interview "Answering in an interview"
    On allocators: do not call a general-purpose allocator on the hot path, `cudaMalloc` / `cudaFree` least of all, since they are slow, synchronize the device and fragment memory; each step's temporary data goes in an arena (allocation moves the offset and everything is reclaimed at once); a fixed-size block allocator makes allocation and freeing O(1) through a free stack with no external fragmentation, and reference counts let several sequences share a block, with copy-on-write before writing to a shared one, which is exactly a paged KV cache's memory management. PyTorch's caching allocator does not return memory to the driver, so reserved exceeds allocated, and fragmentation causes "free memory yet the allocation fails" OOMs (which `expandable_segments` eases). `std::pmr` lets a standard container allocate from a chosen memory resource.

## Exercises {#练习}

1. Write an aligned allocator `AlignedAllocator<T, Align>` so that a `std::vector<float, AlignedAllocator<float, 64>>`'s buffer is 64-byte aligned and `AlignedAllocator<float, 4096>` is page-aligned.

??? success "Answer"
    An allocator needs only `value_type`, `allocate`, `deallocate` and an equality comparison; because the template has a non-type parameter, it also needs `rebind` to tell the container how to get "the same alignment with a different element type". The aligned allocation uses C++17's `operator new(size, std::align_val_t)`:

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
      v.resize(5000);   // still aligned after growing
      std::printf("64 字节对齐：%s，页对齐：%s\n", aligned(v, 64) ? "是" : "否", aligned(page, 4096) ? "是" : "否");
    }
    ```

    ```text title="output"
    64 字节对齐：是，页对齐：是
    ```

2. After a production inference service has run for a while, `torch.cuda.memory_reserved()` is 70 GB and `memory_allocated()` is 50 GB, and allocating a 4 GB tensor reports OOM. Explain why, and give at least two ways to ease it.

??? success "Answer"
    The difference between reserved and allocated (20 GB) is free blocks the caching allocator keeps for reuse, but they are many blocks of varying size and none (nor any mergeable adjacent run) can satisfy a contiguous 4 GB request. That is external fragmentation.
    Ways to ease it:
    settle the shapes down and make fewer requests of differing size (pad to a few fixed batch sizes, pin the temporary buffers down with a CUDA Graph);
    preallocate the large blocks at startup (the KV cache, the workspace) and carve from your own pool at run time;
    turn on PyTorch's `expandable_segments` (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), which lets a segment grow through virtual memory mappings and cuts fragmentation;
    call `torch.cuda.empty_cache()` at an idle moment to return the free blocks to the driver (at the cost of `cudaMalloc`ing again afterwards).

## Summary {#小结}

- [x] Do not call a general-purpose allocator on the hot path, `cudaMalloc` least of all: slow, synchronizing and fragmenting.
- [x] An arena: allocation moves the offset and everything is reclaimed at once, which suits each step's temporary data; only objects needing no destructor go in it.
- [x] A fixed-size block allocator: a free stack gives O(1) allocation and freeing with no fragmentation; reference counts support sharing, with copy-on-write before writing to a shared block, which is paged KV's memory management.
- [x] A caching allocator keeps freed blocks by size for reuse, at the cost of reserved exceeding allocated and of OOMs caused by fragmentation.
- [x] `std::pmr` lets a standard container allocate from a chosen memory resource; a custom allocator provides `allocate`, `deallocate` and an equality comparison, plus `rebind` when the template has a non-type parameter.
