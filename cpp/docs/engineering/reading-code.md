# 读懂推理基础库的 C++

<p class="lead">前面各章的知识点，在真实的推理库里是以什么样子出现的？这一章以 vLLM 0.30.0 的 <code>csrc/</code> 为例，跟着一个算子从 Python 调用一路读到 kernel 启动，再整理一份"模式目录"：看到某种写法，就知道它在解决什么问题、该回到哪一章。最后给出读其他基础库时的路线。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个 `torch.ops._C.xxx` 算子，在 C++ 源码里从哪里开始找？
    2. 算子 schema 里的 `Tensor!` 是什么意思？
    3. 为什么很多 C 接口的错误检查宏只记录错误、不抛异常？
    4. 多张 GPU 之间用"标志"做同步时，为什么写标志用 release、读标志用 acquire？标志为什么要按 128 字节对齐？
    5. 读一个陌生的 kernel 文件，应该先看什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 搜 `TORCH_LIBRARY` 或者 `ops.def` / `ops.impl` 找到注册的地方，看它绑定的 C++ 函数（host 端的入口），再顺着找到它启动的 kernel。
    2. 这个参数会被原地修改（带别名注解的可变张量，例如 `Tensor(a!)`）。
    3. C 接口不能让 C++ 异常穿过边界（调用方可能是 C 或者其他语言，异常穿过去是未定义行为）；而且在 kernel 启动、回调这些位置抛异常不安全，所以只记录错误码，由调用方检查。
    4. 写方用 release：保证之前写的数据先于标志对对方可见；读方用 acquire：看到标志之后才去读数据，不会读到旧值。按 128 字节对齐：GPU 访问内存（包括跨 NVLink 访问对端显存）的粒度比 CPU 的 64 字节缓存行大（L2 的行是 128 字节），不同 rank 写的标志不能落在同一个缓存行里。
    5. 先看入口和注册（它提供了什么接口），再看 host 端：形状、步长、设备和数据类型的检查、launch 配置、派发；然后看所有权、生命周期和同步点；最后才读 kernel 里面的计算细节。

## 跟着一个算子读：`reshape_and_cache`

`reshape_and_cache` 把本步新算出来的 K、V 写进分页 KV Cache 的对应槽位（推理系统手册的[分页 KV Cache](serving://engine/paged-kv/) 里用 Python 实现过同样的事）。它在 vLLM 里的完整路径：

**第一步：注册与 schema。**在 `csrc/libtorch_stable/torch_bindings.cpp` 里：

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

- `def` 声明 schema，`impl` 为 CUDA 设备注册实现——和[上一章](python-binding.md)的 `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` 是同一套机制；
- `Tensor!` 表示这个参数**会被原地修改**。`torch.compile` 靠它知道这个算子有副作用、不能被删掉或重排；返回值是 `()`，结果写在 `key_cache` / `value_cache` 里；
- `STABLE_` 前缀是 PyTorch 的**稳定 ABI**：编译出来的扩展不依赖 PyTorch 内部的 C++ 类布局，换一个 PyTorch 版本不用重新编译。C++ 的 ABI 问题（类的布局、名字修饰、标准库的版本）是二进制分发时最头疼的事，回忆[编译模型](../basics/compile-ub.md)里链接器只认符号名。

**第二步：host 端函数。**在 `csrc/libtorch_stable/cache_kernels.cu` 里（节选）：

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

逐行对应：

- **注释里写形状**：每个张量参数后面都标了形状。读 kernel 相关的代码，第一件事就是把这些形状抄下来——`key_cache` 的 `[num_blocks, num_heads, head_size/x, block_size, x]` 是一种为了让 kernel 按 `x` 个元素一组做向量化读取而设计的布局（[对象布局、对齐与缓存](../memory/layout.md)）；
- **用 `stride(0)` 而不是假设连续**：`key` 可能是从一个更大的 QKV 张量里切出来的视图，行与行之间的步长不等于 `num_heads * head_size`（上一章说的"处理非连续输入"的另一种做法：不 `contiguous()`，而是把步长传进 kernel）；
- **`DeviceGuard`**：构造时把当前 CUDA 设备切换到 `key` 所在的卡，析构时切回去——一个标准的 RAII 守卫（[值语义与 RAII](../basics/value-raii.md)）。多卡进程里忘了切设备，kernel 就会在错误的卡上启动；
- **`get_current_cuda_stream()`**：在 PyTorch 当前的 stream 上启动，才能和前后的算子正确排序，也才能被 CUDA Graph 捕获；
- **`DISPATCH_BY_KV_CACHE_DTYPE`**：运行时的"输入类型 × KV Cache 类型"派发到编译期的模板参数（[模板、concepts 与 constexpr](../basics/templates.md)）。

**第三步：派发宏。**在 `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh` 里（节选）：

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

`FN` 就是上面的 `CALL_RESHAPE_AND_CACHE`，它展开成 `vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE><<<grid, block, 0, stream>>>(...)`。于是 3 种输入类型 × 3 种 KV Cache 类型，一共实例化了若干份 kernel，每份都在编译期确定了"读什么类型、存什么类型、要不要量化"。
注意 fp16 在这里用 `uint16_t` 表示：kernel 只是搬运和转换位模式，不需要 half 的算术——这和[编译模型与未定义行为](../basics/compile-ub.md#严格别名与-stdbit_cast)里用整数类型看浮点位模式是同一个思路。

**第四步：最后才是 kernel。**到这里，每个参数的含义、布局和类型都清楚了，再读 kernel 本身就只剩"每个线程负责哪些元素"的问题（CUDA 手册的内容）。

## 模式目录

| 看到这种写法 | 它在做什么 | vLLM 0.30.0 里的例子 | 回到哪一章 |
| --- | --- | --- | --- |
| `*_DISPATCH_*` 宏、嵌套的 `switch` / lambda | 运行时类型、维度 → 模板参数 | `csrc/dispatch_utils.h`、`DISPATCH_BY_KV_CACHE_DTYPE` | [模板](../basics/templates.md) |
| `*Guard` 对象（`DeviceGuard`、`CUDAGuard`、`CUDAStreamGuard`） | 构造时切换设备 / stream，析构时恢复 | `cache_kernels.cu` 里的 `DeviceGuard` | [RAII](../basics/value-raii.md) |
| `TORCH_CHECK`、`CUDA_CHECK` 宏，外面包一层 `do { ... } while (0)` | 检查条件，失败时抛异常或记录错误；`do-while(0)` 让宏在 `if` / `else` 里也表现得像一条语句 | `cumem_allocator.cpp` 的 `CUDA_CHECK` | [编译模型](../basics/compile-ub.md)、[Python 互操作](python-binding.md) |
| `extern "C" { ... }` 里的函数、全局的回调指针 | 给 C 语言或 `ctypes` / PyTorch 的可插拔分配器调用的接口 | `cumem_allocator.cpp` | [编译模型](../basics/compile-ub.md#名字修饰与-extern-c) |
| `PyGILState_Ensure()` / `PyGILState_Release()` | 在 C++ 里回调 Python 之前拿到 GIL | `cumem_allocator.cpp` 回调 Python 的分配钩子 | [Python 互操作](python-binding.md#gil) |
| `alignas(128)`、`__align__(16)` 的结构体 | 标志独占缓存行避免伪共享；向量化读取要求对齐 | `custom_collective_common.cuh` 的 `Signal`、`RankData` | [布局与缓存](../memory/layout.md) |
| `st.release` / `ld.acquire` 内联汇编、`__threadfence` | 跨 GPU 的"先写数据、再写标志"同步 | `st_flag_release` / `ld_flag_acquire` | [atomic 与内存序](../concurrency/atomics.md) |
| `if constexpr (...)` 出现在 kernel 里 | 按模板参数在编译期裁掉分支 | `template <int ngpus, bool final_sync = false> barrier_at_end` 里的 `if constexpr (!final_sync)` | [模板](../basics/templates.md#if-constexpr编译期分支) |
| 大的预分配缓冲区 + 自己管理偏移 / 块号 | 自己做内存管理，避开通用分配器 | KV Cache 的块、custom all-reduce 的 IPC 缓冲区、`cumem_allocator` 的虚拟内存映射 | [分配器与内存池](../memory/allocators.md) |

### 两个值得细看的片段

**C 接口里的错误处理。**`cumem_allocator.cpp` 是给 PyTorch 的"可插拔分配器"用的 C 接口，它的 `CUDA_CHECK` 只记录错误码和错误信息、打印出来，**不抛异常**：

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

原因是这些函数被 C 调用约定的代码调用，C++ 异常穿过 `extern "C"` 的边界是未定义行为；所以错误被存进全局变量，由 Python 那一侧在调用结束后检查。
同一个文件里回调 Python 的地方手动配对 `PyGILState_Ensure()` 和 `PyGILState_Release()`，每条返回路径上都要记得释放——这正是 RAII 守卫（比如 pybind11 的 `py::gil_scoped_acquire`）要解决的问题：在纯 C 接口里用不了 pybind11 时，就只能靠手工配对和代码审查。

**跨 GPU 的屏障。**vLLM 的 custom all-reduce 在 NVLink 连通的几张卡之间，用互相可见的显存（CUDA IPC）直接读写对方的数据。每张卡有一个 `Signal` 结构，标志按 128 字节对齐；同步时每张卡给每个对端写一个递增的标志值，再等自己收到所有对端的标志：

```cpp
// 节选，省略了 Volta 之前架构的分支
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}
static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}
// barrier_at_start_release 里：
    st_flag_release(peer_counter_ptr, flag);            // 通知对端：我的数据写好了
    while (ld_flag_acquire(self_counter_ptr) != flag);  // 等对端通知我
```

`.sys` 表示这个内存序的作用范围是整个系统（跨 GPU、跨 CPU），`release` / `acquire` 的含义和 C++ 的 `memory_order_release` / `memory_order_acquire` 完全一样。
用 CPU 线程就能把这个协议完整地模拟出来，并用 TSan 验证它没有数据竞争：

```cpp title="flag_barrier_allreduce.cpp" sanitize="thread"
// 用 CPU 线程模拟"写输入 → release 写对端的标志 → acquire 等自己的标志 → 读所有对端的输入"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

constexpr int kRanks = 4, kN = 8;

struct alignas(128) Flag {   // 每个标志独占 128 字节，和 vLLM 的 Signal 一样
  std::atomic<std::uint32_t> v{0};
};

struct Rank {
  float data[kN];            // 本 rank 的输入：相当于 IPC 映射出来、对端可以直接读的显存
  Flag start[kRanks];        // start[p]：rank p 在"开始屏障"写给我的标志
  Flag end[kRanks];          // end[p]：rank p 在"结束屏障"写给我的标志
};

// 两个屏障必须用不同的标志数组（at_end 选择用哪一组）
void barrier(std::vector<Rank>& ranks, bool at_end, int me, std::uint32_t flag) {
  auto slot = [&](int owner, int from) -> Flag& { return at_end ? ranks[owner].end[from] : ranks[owner].start[from]; };
  for (int p = 0; p < kRanks; ++p) slot(p, me).v.store(flag, std::memory_order_release);   // 通知每个对端
  for (int p = 0; p < kRanks; ++p) {
    while (slot(me, p).v.load(std::memory_order_acquire) != flag) {                         // 等每个对端
      std::this_thread::yield();   // GPU 上是空转；CPU 线程可能比核数多，让出时间片
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
        for (int i = 0; i < kN; ++i) ranks[r].data[i] = float(r + 1) * round;   // 写自己的输入
        barrier(ranks, false, r, round);    // 开始屏障：所有人的输入都写好了
        for (int i = 0; i < kN; ++i) {
          float s = 0;
          for (int p = 0; p < kRanks; ++p) s += ranks[p].data[i];              // 直接读对端的输入求和
          out[r][i] = s;
        }
        barrier(ranks, true, r, round);     // 结束屏障：所有人都读完了，下一轮才能覆盖输入
      }
    });
  }
  for (auto& t : ts) t.join();
  std::printf("第 3 轮 all-reduce 的结果：");
  for (int r = 0; r < kRanks; ++r) std::printf("%.0f ", out[r][0]);
  std::printf("\n");
}
```

```text title="输出"
第 3 轮 all-reduce 的结果：30 30 30 30
```

两个细节都能在 vLLM 的 `Signal` 里找到对应：

- **需要结束屏障**：去掉它，快的 rank 会在慢的 rank 读完之前就覆盖自己的输入——TSan 会立刻报告数据竞争。kernel 里对应开头的 `barrier_at_start` 和结尾的 `barrier_at_end`；
- **两个屏障用两组标志**（`Signal` 里的 `start` 和 `end`）：如果共用一组，快的 rank 过了开始屏障就会去写结束屏障的标志，把慢的 rank 还没来得及看到的上一个值覆盖掉，慢的 rank 就永远等不到它要的值——本书写这个例子时的第一版就是这样死锁的。分成两组之后，一个 rank 要再次写 `start`，必须先过结束屏障，而那要求所有人都已经过了开始屏障。

标志值每轮递增而不是在 0 和 1 之间翻转，是为了不需要"复位"这一步：复位本身又需要一次同步。

## 读其他基础库时看什么

| 库 | 读什么 | 用到的知识 |
| --- | --- | --- |
| FlashInfer | 头文件里的模板 kernel、`plan` / `run` 分离的调度结构、按需编译（JIT）的机制 | 模板与派发、编译模型 |
| SGLang 的 `sgl-kernel` | 算子的注册文件、每个算子的 host 端函数（形状检查、派发、stream） | Python 互操作、模板 |
| DeepEP | 管理 NVLink / RDMA 缓冲区的 `Buffer` 类、NVSHMEM 的使用、通信与计算重叠的 hook | RAII、内存序、分配器 |
| Mooncake Transfer Engine | 各种传输后端（RDMA、TCP 等）的抽象接口、内存注册、批量异步传输的提交与完成 | 虚函数与模板、所有权、并发 |
| PyTorch 的 `c10/cuda/CUDACachingAllocator` | 按大小分桶、拆分与合并块、跨 stream 使用的记录 | 分配器与内存池 |

读的方法都一样：

1. **从入口找起**：Python 里的调用 → 注册处（`TORCH_LIBRARY`、`PYBIND11_MODULE`）→ host 端函数 → kernel；
2. **先抄下形状和布局**：注释里的形状、`stride` 的用法，画一张数据布局图；
3. **理清所有权和生命周期**：谁分配、谁释放、在哪个 stream / 线程上；RAII 守卫在哪里；
4. **找同步点**：锁、原子标志、event、屏障，各自保护的是什么数据；
5. **最后才读 kernel 的计算细节**。

配置一遍工程、生成 `compile_commands.json`（[CMake 与调试](build-debug.md)），让 IDE 能跳转定义，效率会高得多。

!!! interview "面试怎么答"
    读源码题：一个 `torch.ops._C.xxx` 算子，从注册处（`TORCH_LIBRARY` 的 schema）找到 host 函数，再到 kernel；schema 里的 `Tensor!` 表示这个参数会被原地修改。host 函数里看形状和步长、设备守卫（`OptionalCUDAGuard` 是 RAII）、当前 stream、按数据类型和常量的派发（模板）；最后才看 kernel。C 接口不能让异常穿过边界，错误用返回码或全局状态传递。多卡之间用标志同步时，写标志用 release、读标志用 acquire，标志按 128 字节对齐，避免和别的数据挤在同一个缓存行里。读陌生的库按入口 → 形状与布局 → 所有权与生命周期 → 同步点 → 计算细节的顺序。

## 练习

1. 在 vLLM 0.30.0 的 `csrc/libtorch_stable/torch_bindings.cpp` 里任选一个你熟悉的算子（比如 `rms_norm`、`silu_and_mul`），按本章的四步写出它的"阅读笔记"：schema、哪些参数会被修改、host 端做了哪些检查和派发、kernel 的模板参数有哪些。

??? success "参考要点"
    以 `rms_norm` 为例（`csrc/libtorch_stable/layernorm_kernels.cu`）：

    - schema 是 `rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) -> ()`：结果写进预先分配好的 `result`（`Tensor!`），`weight` 可以不传（`Tensor?`）；
    - host 端检查：`result` 必须连续；`input` 最后一维不连续时先 `contiguous()`，其余维度用 `stride` 传进 kernel（支持 2～4 维的输入视图）；`weight` 的形状必须和 `input` 的最后一维对得上；
    - 启动配置：token 数少时用大的 block、多时用小的 block 增加并发；开启"batch 不变"模式时固定 block 大小，保证同一个请求在不同 batch 里算出的结果逐位相同；
    - 派发：先按张量的维数（2 / 3 / 4），再按浮点类型（fp32 / fp16 / bf16），再按向量化宽度 `vec_size = gcd(16 / sizeof(scalar_t), hidden_size)`（一次读 16 字节，但不能超过 `hidden_size` 能整除的宽度）；
    - kernel 的模板参数是 `<scalar_t, vec_size, tensor_rank, has_weight>`：有没有权重也在编译期确定。

    对照[上一章](python-binding.md)自己写的 CPU 版 RMSNorm，生产代码多处理了非连续的多维输入、可选的权重、向量化宽度和结果的确定性。

2. 为什么 `Signal` 里的标志要按 128 字节而不是 64 字节对齐？如果所有标志挨在一起，会发生什么？

??? success "参考答案"
    GPU 访问全局内存（以及跨 NVLink 访问对端显存）的粒度比 CPU 的 64 字节缓存行更大（L2 的缓存行是 128 字节），按 128 字节对齐保证不同 rank、不同 block 写的标志不会落在同一个缓存行里。
    挨在一起时，多张卡同时写"各自的"标志会在同一个缓存行上争用（伪共享的跨设备版本），轮询的一方也会因为别人的写入而反复失效、重读，屏障的延迟会明显上升。

## 小结

- [x] 读算子从入口到 kernel：注册与 schema（`Tensor!` 表示原地修改）→ host 端函数（形状、步长、设备守卫、stream、派发）→ 最后才是 kernel。
- [x] 真实代码里的模式都能对应到前面的章节：派发宏是模板，各种 Guard 是 RAII，`alignas` 是布局与伪共享，标志同步是 release / acquire。
- [x] C 接口里不能让异常穿过边界，错误用返回码或全局状态传递；回调 Python 要手动管理 GIL。
- [x] 读任何基础库的顺序：入口 → 形状与布局 → 所有权与生命周期 → 同步点 → 计算细节。
