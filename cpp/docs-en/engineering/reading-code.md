# Reading an inference library's C++

<p class="lead">What do the earlier chapters' topics look like in a real inference library? This chapter takes vLLM 0.30.0's <code>csrc/</code> and follows one operator from the Python call all the way to the kernel launch, then assembles a "catalogue of patterns": seeing a construct, you know what problem it solves and which chapter to go back to. It ends with a route for reading other libraries.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Where do you start looking in the C++ source for a `torch.ops._C.xxx` operator?
    2. What does `Tensor!` mean in an operator's schema?
    3. Why do many error-checking macros on a C interface only record the error rather than throwing?
    4. When several GPUs synchronize through a flag, why is the write a release and the read an acquire? Why is the flag aligned to 128 bytes?
    5. Reading an unfamiliar kernel file, what do you look at first?

??? success "Answers (try it yourself first, then expand)"
    1. Search for `TORCH_LIBRARY` or `ops.def` / `ops.impl` to find the registration, look at the C++ function it binds (the host-side entry point), and follow that to the kernel it launches.
    2. That argument is modified in place (a mutable tensor with an alias annotation, `Tensor(a!)` say).
    3. A C interface must not let a C++ exception cross the boundary (the caller may be C or another language, where an exception crossing is undefined behaviour); and throwing at a kernel launch or inside a callback is unsafe, so only an error code is recorded for the caller to check.
    4. The writer uses release so that the data written before is visible to the other side before the flag; the reader uses acquire so that it reads the data only after seeing the flag and never reads a stale value. Aligned to 128 bytes: a GPU's memory access granularity (including reaching the peer's device memory over NVLink) is larger than a CPU's 64-byte cache line (the L2 line is 128 bytes), so flags written by different ranks must not land in the same line.
    5. The entry point and the registration first (what interface it offers), then the host side: the checks on shapes, strides, device and data type, the launch configuration, the dispatch; then the ownership, lifetimes and synchronization points; and only last the computation inside the kernel.

## Following one operator: `reshape_and_cache` {#跟着一个算子读reshape_and_cache}

`reshape_and_cache` writes the K and V just computed into the matching slots of a paged KV cache (the Inference Systems handbook's [paged KV cache](serving://engine/paged-kv/) implemented the same thing in Python). Its full path in vLLM:

**Step one: the registration and the schema.** In `csrc/libtorch_stable/torch_bindings.cpp`:

```cpp
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  // ...
  ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  // ...
  ops.impl("reshape_and_cache", TORCH_BOX(&reshape_and_cache));
}
```

- `def` declares the schema and `impl` registers the implementation for the CUDA device, the same mechanism as [the previous chapter's](python-binding.md) `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`;
- `Tensor!` says this argument **is modified in place**. That is how `torch.compile` knows the operator has side effects and must not be deleted or reordered; the return type is `()` and the results go into `key_cache` / `value_cache`;
- the `STABLE_` prefix is PyTorch's **stable ABI**: the compiled extension does not depend on the layout of PyTorch's internal C++ classes, so a different PyTorch version needs no rebuild. C++'s ABI problems (class layout, name mangling, the standard library's version) are the greatest headache in distributing binaries; recall from [the compilation model](../basics/compile-ub.md) that the linker knows only symbol names.

**Step two: the host-side function.** In `csrc/libtorch_stable/cache_kernels.cu` (excerpted):

```cpp
void reshape_and_cache(
    torch::stable::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::stable::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::stable::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::stable::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::stable::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  int head_div_x = head_size / x;

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_div_x, 512));
  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.scalar_type(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE);
}
```

Line by line:

- **the shapes are in the comments**: every tensor parameter has its shape beside it. Reading kernel-related code, the first thing to do is copy those shapes down; `key_cache`'s `[num_blocks, num_heads, head_size/x, block_size, x]` is a layout designed so the kernel can read vectorized groups of `x` elements ([object layout, alignment and the cache](../memory/layout.md));
- **using `stride(0)` rather than assuming contiguity**: `key` may be a view sliced out of a larger QKV tensor, where the stride between rows is not `num_heads * head_size` (the other approach to "handling non-contiguous inputs" from the previous chapter: pass the strides into the kernel instead of calling `contiguous()`);
- **`DeviceGuard`**: switches the current CUDA device to the card `key` is on at construction and switches back at destruction, a standard RAII guard ([value semantics and RAII](../basics/value-raii.md)). Forget to switch in a multi-GPU process and the kernel launches on the wrong card;
- **`get_current_cuda_stream()`**: launching on PyTorch's current stream is what orders it correctly against the operators around it and what lets a CUDA Graph capture it;
- **`DISPATCH_BY_KV_CACHE_DTYPE`**: a run-time "input type × KV cache type" dispatched to compile-time template arguments ([templates, concepts and constexpr](../basics/templates.md)).

**Step three: the dispatch macro.** In `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh` (excerpted):

```cpp
#define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
    vllm::Fp8KVCacheDataType KV_CACHE_DTYPE =                                  \
        vllm::get_fp8_kv_cache_data_type(KV_DTYPE);                            \
    if (KV_CACHE_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {                   \
      if (SRC_DTYPE == torch::headeronly::ScalarType::Float) {                 \
        FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                     \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::Half) {           \
        FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);               \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::BFloat16) {       \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);     \
      } else { /* 报错 */ }                                                    \
    } else if (KV_CACHE_DTYPE == vllm::Fp8KVCacheDataType::kFp8E4M3) {         \
      if (SRC_DTYPE == torch::headeronly::ScalarType::Float) {                 \
        FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);                \
      } /* ... */
```

`FN` is the `CALL_RESHAPE_AND_CACHE` above, expanding into `vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE><<<grid, block, 0, stream>>>(...)`. So 3 input types × 3 KV cache types instantiate a number of kernels, each with "which type to read, which to store, whether to quantize" settled at compile time.
Note that fp16 is represented by `uint16_t` here: the kernel only moves and converts bit patterns and needs no half arithmetic, the same idea as looking at a floating-point bit pattern through an integer type in [the compilation model and undefined behaviour](../basics/compile-ub.md#严格别名与-stdbit_cast).

**Step four: the kernel, last.** By now every parameter's meaning, layout and type is clear, and reading the kernel leaves only "which elements each thread handles" (the CUDA handbook's material).

## A catalogue of patterns {#模式目录}

| Seeing this | What it does | An example in vLLM 0.30.0 | Which chapter |
| --- | --- | --- | --- |
| a `*_DISPATCH_*` macro, nested `switch` / lambdas | run-time types and dimensions into template arguments | `csrc/dispatch_utils.h`, `DISPATCH_BY_KV_CACHE_DTYPE` | [templates](../basics/templates.md) |
| a `*Guard` object (`DeviceGuard`, `CUDAGuard`, `CUDAStreamGuard`) | switches the device / stream on construction and restores it on destruction | the `DeviceGuard` in `cache_kernels.cu` | [RAII](../basics/value-raii.md) |
| the `TORCH_CHECK` and `CUDA_CHECK` macros, wrapped in a `do { ... } while (0)` | check a condition and throw or record an error on failure; the `do-while(0)` makes the macro behave like one statement inside an `if` / `else` | `cumem_allocator.cpp`'s `CUDA_CHECK` | [the compilation model](../basics/compile-ub.md), [Python interop](python-binding.md) |
| functions inside `extern "C" { ... }`, global callback pointers | an interface for C, `ctypes` or PyTorch's pluggable allocator to call | `cumem_allocator.cpp` | [the compilation model](../basics/compile-ub.md#名字修饰与-extern-c) |
| `PyGILState_Ensure()` / `PyGILState_Release()` | taking the GIL before calling back into Python from C++ | `cumem_allocator.cpp`'s callbacks into Python's allocation hooks | [Python interop](python-binding.md#gil) |
| structs with `alignas(128)` or `__align__(16)` | a flag gets a cache line to itself to avoid false sharing; vectorized reads require alignment | `custom_collective_common.cuh`'s `Signal` and `RankData` | [layout and the cache](../memory/layout.md) |
| `st.release` / `ld.acquire` inline assembly, `__threadfence` | cross-GPU "write the data, then write the flag" synchronization | `st_flag_release` / `ld_flag_acquire` | [atomics and memory order](../concurrency/atomics.md) |
| `if constexpr (...)` inside a kernel | cutting a branch at compile time by a template argument | the `if constexpr (!final_sync)` in `template <int ngpus, bool final_sync = false> barrier_at_end` | [templates](../basics/templates.md#if-constexpr编译期分支) |
| a large preallocated buffer with offsets or block numbers managed by hand | managing memory directly, avoiding the general-purpose allocator | the KV cache's blocks, custom all-reduce's IPC buffers, `cumem_allocator`'s virtual memory mappings | [allocators and memory pools](../memory/allocators.md) |

### Two passages worth a closer look {#两个值得细看的片段}

**Error handling in a C interface.** `cumem_allocator.cpp` is the C interface for PyTorch's "pluggable allocator", and its `CUDA_CHECK` only records the error code and message and prints them, **throwing nothing**:

```cpp
#define CUDA_CHECK(condition)                                           \
  do {                                                                  \
    CUresult error = condition;                                         \
    if (error != 0) {                                                   \
      error_code = error;                                               \
      char* error_string;                                               \
      cuGetErrorString(error, (const char**)&error_string);             \
      snprintf(error_msg, sizeof(error_msg), "CUDA Error: %s at %s:%d", \
               error_string, __FILE__, __LINE__);                       \
      std::cerr << error_msg << std::endl;                              \
    }                                                                   \
  } while (0)
```

The reason is that these functions are called through the C calling convention, and a C++ exception crossing an `extern "C"` boundary is undefined behaviour; so the error goes into a global variable for the Python side to check once the call returns.
Elsewhere in the same file, the calls back into Python pair `PyGILState_Ensure()` with `PyGILState_Release()` by hand, which every return path has to remember, which is exactly what an RAII guard (pybind11's `py::gil_scoped_acquire`) solves: where pybind11 cannot be used in a pure C interface, nothing is left but manual pairing and code review.

**A cross-GPU barrier.** vLLM's custom all-reduce reads and writes the other cards' data directly in mutually visible device memory (CUDA IPC) across the cards connected by NVLink. Each card has a `Signal` struct whose flags are aligned to 128 bytes; to synchronize, each card writes an incrementing flag value to every peer and then waits until it has received flags from all of them:

```cpp
// excerpted, with the pre-Volta branch omitted
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}
static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}
// inside barrier_at_start_release:
    st_flag_release(peer_counter_ptr, flag);            // tell the peer: my data is written
    while (ld_flag_acquire(self_counter_ptr) != flag);  // wait for the peer to tell me
```

`.sys` says this memory order's scope is the whole system (across GPUs and the CPU), and `release` / `acquire` mean exactly what C++'s `memory_order_release` / `memory_order_acquire` mean.
The protocol can be simulated completely with CPU threads and verified free of data races with TSan:

```cpp title="flag_barrier_allreduce.cpp" sanitize="thread"
// CPU threads simulating "write the input -> write the peers' flags with release -> wait on my own flags with acquire -> read every peer's input"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

constexpr int kRanks = 4, kN = 8;

struct alignas(128) Flag {   // each flag gets 128 bytes to itself, as in vLLM's Signal
  std::atomic<std::uint32_t> v{0};
};

struct Rank {
  float data[kN];            // this rank's input: standing in for IPC-mapped device memory the peers can read directly
  Flag start[kRanks];        // start[p]: the flag rank p writes to me at the start barrier
  Flag end[kRanks];          // end[p]: the flag rank p writes to me at the end barrier
};

// the two barriers have to use different flag arrays (at_end picks which)
void barrier(std::vector<Rank>& ranks, bool at_end, int me, std::uint32_t flag) {
  auto slot = [&](int owner, int from) -> Flag& { return at_end ? ranks[owner].end[from] : ranks[owner].start[from]; };
  for (int p = 0; p < kRanks; ++p) slot(p, me).v.store(flag, std::memory_order_release);   // tell every peer
  for (int p = 0; p < kRanks; ++p) {
    while (slot(me, p).v.load(std::memory_order_acquire) != flag) {                         // wait for every peer
      std::this_thread::yield();   // on a GPU this spins; there may be more CPU threads than cores, so yield the time slice
    }
  }
}

int main() {
  std::vector<Rank> ranks(kRanks);
  std::vector<std::vector<float>> out(kRanks, std::vector<float>(kN));
  std::vector<std::thread> ts;
  for (int r = 0; r < kRanks; ++r) {
    ts.emplace_back([&, r] {
      for (std::uint32_t round = 1; round <= 3; ++round) {
        for (int i = 0; i < kN; ++i) ranks[r].data[i] = float(r + 1) * round;   // write our own input
        barrier(ranks, false, r, round);    // the start barrier: everyone's input is written
        for (int i = 0; i < kN; ++i) {
          float s = 0;
          for (int p = 0; p < kRanks; ++p) s += ranks[p].data[i];              // read the peers' inputs directly and sum
          out[r][i] = s;
        }
        barrier(ranks, true, r, round);     // the end barrier: everyone has finished reading, so the next round may overwrite the inputs
      }
    });
  }
  for (auto& t : ts) t.join();
  std::printf("第 3 轮 all-reduce 的结果：");
  for (int r = 0; r < kRanks; ++r) std::printf("%.0f ", out[r][0]);
  std::printf("\n");
}
```

```text title="output"
第 3 轮 all-reduce 的结果：30 30 30 30
```

Both details correspond to something in vLLM's `Signal`:

- **the end barrier is necessary**: without it, a fast rank overwrites its own input before a slow rank has finished reading it, and TSan reports the data race at once. In the kernel this is the `barrier_at_start` at the beginning and the `barrier_at_end` at the end;
- **the two barriers use two sets of flags** (`start` and `end` in `Signal`): with one shared set, a fast rank past the start barrier would write the end barrier's flag and overwrite a value the slow rank had not seen yet, leaving the slow rank waiting forever for the value it needs; the first version written for this example deadlocked exactly that way. With two sets, a rank writing `start` again has to pass the end barrier first, which requires everyone to have passed the start barrier.

The flag value increments each round rather than toggling between 0 and 1 so that no "reset" step is needed: a reset would itself need a synchronization.

## What to look at in other libraries {#读其他基础库时看什么}

| Library | What to read | What it draws on |
| --- | --- | --- |
| FlashInfer | the template kernels in the headers, the `plan` / `run` scheduling split, the on-demand compilation (JIT) machinery | templates and dispatch, the compilation model |
| SGLang's `sgl-kernel` | the operator registration files, each operator's host-side function (shape checks, dispatch, the stream) | Python interop, templates |
| DeepEP | the `Buffer` class managing the NVLink / RDMA buffers, the use of NVSHMEM, the hooks that overlap communication with computation | RAII, memory order, allocators |
| Mooncake Transfer Engine | the abstract interface over the transport back ends (RDMA, TCP and so on), memory registration, submitting and completing batched asynchronous transfers | virtual functions and templates, ownership, concurrency |
| PyTorch's `c10/cuda/CUDACachingAllocator` | bucketing by size, splitting and merging blocks, recording cross-stream use | allocators and memory pools |

The method is always the same:

1. **start from the entry point**: the Python call → the registration (`TORCH_LIBRARY`, `PYBIND11_MODULE`) → the host-side function → the kernel;
2. **copy the shapes and layouts down first**: the shapes in the comments and the uses of `stride`, drawn into a layout diagram;
3. **work out the ownership and lifetimes**: who allocates, who frees, on which stream or thread; where the RAII guards are;
4. **find the synchronization points**: locks, atomic flags, events, barriers, and what data each protects;
5. **read the kernel's computation last**.

Configuring the project once to generate `compile_commands.json` ([CMake and debugging](build-debug.md)), so the IDE can jump to definitions, makes all of this far quicker.

!!! interview "Answering in an interview"
    On reading source: for a `torch.ops._C.xxx` operator, go from the registration (`TORCH_LIBRARY`'s schema) to the host function and then to the kernel; `Tensor!` in the schema says the argument is modified in place. In the host function, look at the shapes and strides, the device guard (`OptionalCUDAGuard` is RAII), the current stream and the dispatch by data type and constants (templates); the kernel comes last. A C interface must not let an exception cross the boundary and passes errors through a return code or global state. When cards synchronize through a flag, the write is a release and the read an acquire, and the flag is aligned to 128 bytes so it does not share a cache line with anything else. Read an unfamiliar library in the order entry point → shapes and layout → ownership and lifetimes → synchronization points → the computation.

## Exercises {#练习}

1. Pick an operator you know in vLLM 0.30.0's `csrc/libtorch_stable/torch_bindings.cpp` (`rms_norm` or `silu_and_mul`, say) and write its "reading notes" by this chapter's four steps: the schema, which arguments are modified, what checks and dispatch the host side does, and what template arguments the kernel takes.

??? success "The key points"
    Taking `rms_norm` (`csrc/libtorch_stable/layernorm_kernels.cu`):

    - the schema is `rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) -> ()`: the result goes into the preallocated `result` (`Tensor!`) and `weight` is optional (`Tensor?`);
    - the host's checks: `result` has to be contiguous; when `input`'s last dimension is not contiguous it is made `contiguous()` first and the other dimensions' `stride`s are passed into the kernel (supporting 2 to 4 dimensional input views); `weight`'s shape has to match `input`'s last dimension;
    - the launch configuration: a large block for few tokens and a small one for many, to raise the concurrency; in "batch invariant" mode the block size is fixed, so one request gives bit-identical results in different batches;
    - the dispatch: by the tensor's rank (2 / 3 / 4), then by floating-point type (fp32 / fp16 / bf16), then by the vectorization width `vec_size = gcd(16 / sizeof(scalar_t), hidden_size)` (16 bytes per read, but no wider than what divides `hidden_size`);
    - the kernel's template arguments are `<scalar_t, vec_size, tensor_rank, has_weight>`: whether there is a weight is settled at compile time too.

    Against the CPU RMSNorm written in [the previous chapter](python-binding.md), the production code additionally handles non-contiguous multidimensional inputs, an optional weight, the vectorization width and the determinism of the result.

2. Why are `Signal`'s flags aligned to 128 bytes rather than 64? What would happen if all the flags sat next to each other?

??? success "Answer"
    A GPU's granularity for global memory (and for reaching a peer's device memory over NVLink) is larger than a CPU's 64-byte cache line (the L2 line is 128 bytes), and aligning to 128 bytes guarantees that flags written by different ranks and different blocks never land in the same line.
    Side by side, several cards writing "their own" flags would contend on one cache line (false sharing's cross-device version), and the polling side would keep being invalidated and rereading because of other ranks' writes, so the barrier's latency would rise markedly.

## Summary {#小结}

- [x] Read an operator from the entry point to the kernel: the registration and the schema (`Tensor!` meaning modified in place) → the host-side function (shapes, strides, the device guard, the stream, the dispatch) → the kernel last.
- [x] The patterns in real code all map onto the earlier chapters: a dispatch macro is templates, the various Guards are RAII, `alignas` is layout and false sharing, and flag synchronization is release / acquire.
- [x] A C interface must not let an exception cross the boundary and passes errors through a return code or global state; calling back into Python means managing the GIL by hand.
- [x] The order for reading any library: the entry point → shapes and layout → ownership and lifetimes → synchronization points → the computation.
