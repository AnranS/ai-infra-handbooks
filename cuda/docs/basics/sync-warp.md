# 同步、原子操作与 warp 编程

<p class="lead">线程之间需要协作时，就会用到同步和原子操作。这一章讲 block 内同步的规则、原子操作的正确用法与性能代价，以及 warp 级原语：shuffle、vote 和 cooperative groups。warp shuffle 是归约、softmax、attention 等几乎所有高性能算子的基础构件。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 把 `__syncthreads()` 写在 `if (threadIdx.x < 16)` 里面会怎样？
    2. 1000 万个线程对同一个全局变量做 `atomicAdd`，为什么慢？怎么改进？
    3. `__shfl_down_sync(0xffffffff, v, 16)` 做了什么？第一个参数是什么意思？
    4. 怎么用 5 次 shuffle 求出一个 warp 内 32 个数的和？怎么让 32 个线程都拿到这个和？
    5. 不同 block 之间能同步吗？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 只有前 16 个线程执行到 `__syncthreads()`，其他线程永远不到，屏障等不齐：结果是未定义行为（可能死锁，也可能静默出错）。屏障必须被 block 里所有线程执行。
    2. 所有线程争用同一个地址，原子操作只能一个一个地串行完成。先在 warp 内用 shuffle 归约、再在 block 内用共享内存归约，最后每个 block 只做一次全局原子加。
    3. 每个线程取出下标比自己大 16 的那个 lane 的 `v`（超出 warp 范围的保持自己的值）；第一个参数是参与的线程掩码，`0xffffffff` 表示 32 个线程都参与。
    4. 用 `__shfl_down_sync` 依次取偏移 16、8、4、2、1 的值相加，5 次之后 lane 0 得到总和；想让 32 个线程都拿到，改用 `__shfl_xor_sync`（蝶形交换），或者最后再用 `__shfl_sync` 从 lane 0 广播。
    5. kernel 里不能用普通的方法：不同 block 可能根本不同时在运行，互相等待会死锁。要么拆成两个 kernel（kernel 边界就是全局同步），要么用 cooperative launch 的 grid 同步，或者线程块集群（同一个集群内的 block 可以同步）。

## block 内同步：`__syncthreads()`

`__syncthreads()` 是一个**屏障**：block 内所有线程都到达这里之后，才会继续往下执行；并且屏障之前对共享内存和全局内存的写入，对屏障之后同一 block 内的所有线程可见。

最常见的模式是"写共享内存 → 同步 → 读别人写的数据"：

```cuda
tile[threadIdx.x] = input[i];     // 每个线程写一个元素
__syncthreads();                   // 确保所有线程都写完了
float v = tile[blockDim.x - 1 - threadIdx.x];   // 读取别的线程写的元素
```

**两条铁律：**

1. **block 内所有线程都必须执行到同一个 `__syncthreads()`。** 把它放在只有部分线程进入的分支里，行为是未定义的，通常表现为死锁或结果错误：

    ```cuda
    if (threadIdx.x < 16) {
      __syncthreads();   // 错误：其他线程永远不会到达
    }
    ```

    提前 `return` 也会造成同样的问题。处理边界时，让越界的线程"不做事但仍然参与同步"，而不是直接返回。

2. **循环里复用共享内存时，读完之后也要同步。** 否则下一轮的写入可能覆盖别的线程还没读完的数据。GEMM 的分块循环里有两个 `__syncthreads()` 就是这个原因，见 [GEMM](../kernels/gemm.md)。

用 `compute-sanitizer --tool racecheck` 和 `--tool synccheck` 可以检查共享内存的数据竞争和同步错误。

## 原子操作

多个线程更新同一个内存位置时，普通的读-改-写会互相覆盖。原子操作保证"读-改-写"不可分割：

```cuda
atomicAdd(&counter, 1);            // 返回修改前的值
atomicAdd(&sum, x);                // float 也支持
atomicMax(&m, v);                  // int / unsigned
atomicCAS(&addr, compare, val);    // 比较并交换，可以用来实现任意原子操作
atomicExch(&addr, val);
```

支持的类型：`int`、`unsigned`、`unsigned long long` 支持全部操作；`float`、`double` 支持 `atomicAdd`；`__half2`、`__nv_bfloat162` 也有 `atomicAdd`。其他组合（比如 float 的原子 max）需要用 `atomicCAS` 自己实现，见本章练习。

### 原子操作的性能

原子操作本身在 L2 上执行，吞吐不低。慢的原因是**争用**：大量线程更新同一个地址时，这些操作只能串行完成。改进的套路是**分层聚合**：

1. 先在 warp 内用 shuffle 归约，32 个值变成 1 个；
2. 再在 block 内通过共享内存（或共享内存原子操作）归约；
3. 最后每个 block 只做一次全局原子操作。

这样全局原子操作的次数减少到 block 数量级，争用基本消失。直方图是另一个典型例子：先在共享内存里维护每个 block 私有的直方图，最后再合并到全局，见下面的 `histogram.cu`。

!!! warning "浮点原子加法的结果不确定"
    浮点加法不满足结合律，原子操作的执行顺序每次都可能不同，所以用 `atomicAdd` 做浮点求和，**每次运行的结果可能在最后几位不同**。训练需要逐位可复现时，要改用确定性的归约顺序。

## warp 级原语

![图：warp 分歧——同一个 warp 里的两条路径先后执行，各自只有一部分 lane 活跃](../assets/figures/warp-divergence.svg){.aig-svg}

同一个 warp 的线程可以不经过共享内存，直接读取彼此的寄存器，这就是 **shuffle**：

```cuda
T __shfl_sync(unsigned mask, T var, int srcLane, int width = 32);      // 读 srcLane 的 var
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = 32);   // 读 lane - delta
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = 32); // 读 lane + delta
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = 32);    // 读 lane ^ laneMask
```

第一个参数 `mask` 是**参与本次操作的线程掩码**，第 k 位为 1 表示 lane k 参与。整个 warp 都参与时写 `0xffffffff`。所有在 mask 中的线程都必须执行这条指令。shuffle 比经过共享内存交换数据更快，也不需要 `__syncthreads()`。

### warp 归约

![图：warp 归约——__shfl_down_sync 每一步把 offset 之外的值加过来](../assets/figures/warp-reduce.svg){.aig-svg}

用 `__shfl_down_sync` 做 5 次"折半相加"，lane 0 得到 32 个数的和：

```cuda
__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset /= 2)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;   // 只有 lane 0 的结果是完整的和
}
```

改用 `__shfl_xor_sync`，每一步两两交换（蝶形归约），**所有 lane 都得到完整的和**。softmax、LayerNorm 里每个线程都需要用到归约结果，所以更常用这个版本：

```cuda
__device__ __forceinline__ float warp_allreduce_sum(float v) {
  for (int mask = 16; mask > 0; mask /= 2)
    v += __shfl_xor_sync(0xffffffff, v, mask);
  return v;   // 所有 lane 的结果相同
}
```

把 `+` 换成 `fmaxf` 就是求最大值。这两个函数在后面的章节会反复出现。

### 投票与掩码

```cuda
unsigned b = __ballot_sync(0xffffffff, pred);   // 第 k 位 = lane k 的 pred
bool any  = __any_sync(0xffffffff, pred);        // 有任意一个为真
bool all  = __all_sync(0xffffffff, pred);        // 全部为真
int  cnt  = __popc(b);                           // 数 1 的个数
unsigned active = __activemask();                // 当前实际在执行的线程
__syncwarp();                                    // warp 内同步
```

`__ballot_sync` 配合 `__popc` 可以在 warp 内完成"数一数有多少线程满足条件、我前面有几个"，这是流压缩（stream compaction）的基础，见[前缀和](../kernels/scan.md)。

### 一个完整的例子：两级归约求和

```cuda title="block_sum.cu"
// block_sum.cu —— warp shuffle + 共享内存的两级归约，每个 block 一次全局原子加
// 编译：nvcc -O3 -arch=sm_75 block_sum.cu -o block_sum
#include "common.cuh"

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

// 要求 blockDim.x 是 32 的倍数，且不超过 1024
__device__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  v = warp_reduce_sum(v);                    // 第一级：warp 内
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  const int num_warps = blockDim.x / 32;
  v = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
  if (warp == 0) v = warp_reduce_sum(v);     // 第二级：第 0 个 warp 汇总各 warp 的结果
  return v;                                  // 只有线程 0 的值是整个 block 的和
}

__global__ void sum_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) v += x[i];
  v = block_reduce_sum(v);
  if (threadIdx.x == 0) atomicAdd(out, v);   // 每个 block 只做一次原子操作
}

int main() {
  const int n = 50'000'000;
  std::vector<float> h(n);
  fill_random(h, 7, 0.f, 1.f);
  double ref = 0.0;
  for (float v : h) ref += v;

  float *d_x, *d_out;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256, blocks = sm_count() * 4;
  auto run = [&] {
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
    sum_kernel<<<blocks, threads>>>(d_x, d_out, n);
  };
  run();
  CUDA_CHECK_LAST();
  float got = 0.f;
  CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  float ref_f = static_cast<float>(ref);
  bool ok = check_close(&got, &ref_f, 1, 1e-4f, 0.f);   // float 累加有舍入误差，允许 1e-4 的相对误差

  float ms = time_ms(run);
  std::printf("sum of %d floats: %.3f ms, %.1f GB/s\n", n, ms, gbps(n * sizeof(float), ms));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

这里每个线程先用 grid-stride loop 累加自己负责的多个元素（这一步没有任何同步），再做 block 内归约。每个线程先串行累加很多元素，是归约 kernel 能跑满带宽的关键，下一章 [归约](../kernels/reduction.md) 会详细分析。

## 直方图：共享内存私有化

```cuda title="histogram.cu"
// histogram.cu —— 256 个桶的直方图：全局原子 vs 共享内存私有直方图
// 编译：nvcc -O3 -arch=sm_75 histogram.cu -o histogram
#include "common.cuh"
#include <cstdint>

constexpr int kBins = 256;

__global__ void hist_global(const uint8_t* __restrict__ data, int n, unsigned* __restrict__ hist) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    atomicAdd(&hist[data[i]], 1u);
}

__global__ void hist_shared(const uint8_t* __restrict__ data, int n, unsigned* __restrict__ hist) {
  __shared__ unsigned local[kBins];
  for (int b = threadIdx.x; b < kBins; b += blockDim.x) local[b] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    atomicAdd(&local[data[i]], 1u);           // 共享内存原子操作，只在本 block 内争用
  __syncthreads();
  for (int b = threadIdx.x; b < kBins; b += blockDim.x)
    if (local[b]) atomicAdd(&hist[b], local[b]);   // 每个 block 每个桶最多一次全局原子操作
}

int main() {
  const int n = 1 << 26;
  std::vector<uint8_t> h(n);
  std::mt19937 gen(3);
  std::normal_distribution<float> dist(128.f, 20.f);   // 集中在中间的桶，争用更激烈
  for (auto& v : h) v = static_cast<uint8_t>(std::clamp(dist(gen), 0.f, 255.f));
  std::vector<unsigned> ref(kBins, 0), got(kBins);
  for (auto v : h) ref[v]++;

  uint8_t* d_data;
  unsigned* d_hist;
  CUDA_CHECK(cudaMalloc(&d_data, n));
  CUDA_CHECK(cudaMalloc(&d_hist, kBins * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(d_data, h.data(), n, cudaMemcpyHostToDevice));
  const int threads = 256, blocks = sm_count() * 8;

  bool ok = true;
  auto check = [&](const char* name) {
    CUDA_CHECK(cudaMemcpy(got.data(), d_hist, kBins * sizeof(unsigned), cudaMemcpyDeviceToHost));
    bool same = got == ref;
    std::printf("%s: %s\n", name, same ? "PASS" : "FAIL");
    ok &= same;
  };
  CUDA_CHECK(cudaMemset(d_hist, 0, kBins * sizeof(unsigned)));
  hist_global<<<blocks, threads>>>(d_data, n, d_hist);
  CUDA_CHECK_LAST();
  check("hist_global");
  CUDA_CHECK(cudaMemset(d_hist, 0, kBins * sizeof(unsigned)));
  hist_shared<<<blocks, threads>>>(d_data, n, d_hist);
  CUDA_CHECK_LAST();
  check("hist_shared");

  float t1 = time_ms([&] { hist_global<<<blocks, threads>>>(d_data, n, d_hist); });
  float t2 = time_ms([&] { hist_shared<<<blocks, threads>>>(d_data, n, d_hist); });
  std::printf("global atomics: %.3f ms (%.1f GB/s)\nshared atomics: %.3f ms (%.1f GB/s)\n",
              t1, gbps(n, t1), t2, gbps(n, t2));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_hist));
  return ok ? 0 : 1;
}
```

整数直方图的结果是精确的，所以直接比较是否完全相等。进一步的优化方向：每个线程一次读 4 个或 16 个字节；数据高度集中时，在一个 block 内维护多份直方图副本来分散争用。

## Cooperative Groups

Cooperative Groups 是 CUDA 提供的一套更结构化的线程组 API，把"一组线程"抽象成对象，用它写的代码意图更清楚：

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void k(const float* x, float* out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  float v = x[block.group_index().x * block.size() + block.thread_rank()];
  float s = cg::reduce(warp, v, cg::plus<float>());   // warp 内归约，所有线程都拿到结果
  if (warp.thread_rank() == 0) atomicAdd(out, s);
  block.sync();                                         // 等价于 __syncthreads()
}
```

它还支持更小的 tile（`tiled_partition<16>` 等，适合一个 warp 处理多行的小矩阵）、按条件分组（`coalesced_threads()`），以及**整个 grid 的同步**：`cg::this_grid().sync()`。grid 同步要求用 `cudaLaunchCooperativeKernel` 启动，并且所有 block 必须能同时驻留在 GPU 上，否则会死锁。

**不同 block 之间原则上不应该互相等待**（除了上面这种受控的 grid 同步）。需要全局同步时，最常见的做法是把计算拆成两个 kernel。Hopper 的**线程块集群**提供了一个折中：同一个集群里的 block 可以同步并访问彼此的共享内存，见 [Hopper](../advanced/async-hopper.md)。

!!! interview "面试怎么答"
    同步与 warp 编程的常见考点：`__syncthreads()` 必须被 block 里所有线程执行，写在分支里会死锁或出错；原子操作慢在争用，先在 warp、block 内聚合，再做少量全局原子操作，浮点原子加的结果不可复现；shuffle 直接交换寄存器，5 次 `__shfl_xor_sync` 的蝶形归约让 32 个 lane 都拿到和。不同 block 之间不要互相等待，需要全局同步就拆成两个 kernel，或者用 cooperative launch、线程块集群。

## 练习

**1. 浮点原子最大值。** CUDA 没有 float 版本的 `atomicMax`。用 `atomicCAS` 实现 `atomicMaxFloat(float* addr, float val)`。

??? success "参考答案"
    ```cuda title="atomic_max_float.cu"
    // atomic_max_float.cu —— 用 atomicCAS 实现浮点原子最大值
    // 编译：nvcc -O3 -arch=sm_75 atomic_max_float.cu -o atomic_max_float
    #include "common.cuh"

    __device__ float atomicMaxFloat(float* addr, float val) {
      int* p = reinterpret_cast<int*>(addr);
      int old = *p;
      while (__int_as_float(old) < val) {
        int assumed = old;
        old = atomicCAS(p, assumed, __float_as_int(val));   // 成功时返回 assumed
        if (old == assumed) break;
      }
      return __int_as_float(old);
    }

    __global__ void max_kernel(const float* x, int n, float* out) {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicMaxFloat(out, x[i]);
    }

    int main() {
      const int n = 1 << 22;
      std::vector<float> h(n);
      fill_random(h, 11, -100.f, 100.f);
      float ref = *std::max_element(h.begin(), h.end());

      float *d_x, *d_out;
      CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
      float init = -INFINITY;
      CUDA_CHECK(cudaMemcpy(d_out, &init, sizeof(float), cudaMemcpyHostToDevice));
      max_kernel<<<sm_count() * 4, 256>>>(d_x, n, d_out);
      CUDA_CHECK_LAST();
      float got;
      CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
      return check_close(&got, &ref, 1, 0.f, 0.f) ? 0 : 1;
    }
    ```

    循环的含义是：只要当前值比 `val` 小，就尝试把它替换为 `val`；如果在读和换之间被别的线程改了，`atomicCAS` 返回的新值会进入下一轮判断。实际使用时，应该先在 warp 和 block 内求出最大值，再做一次原子操作。

    另一种常见技巧是利用 IEEE 浮点的位模式：非负 float 按位解释成 int 后大小顺序不变，所以非负数可以直接用 `atomicMax((int*)addr, __float_as_int(val))`；负数需要反过来用 `atomicMin` 处理 unsigned 表示。

**2. warp 内的前缀和。** 用 `__shfl_up_sync` 写一个函数，返回 warp 内每个 lane 的包含式前缀和（lane k 得到 lane 0 到 k 的值之和）。

??? success "参考答案"
    ```cuda
    __device__ __forceinline__ int warp_inclusive_scan(int v) {
      const int lane = threadIdx.x % 32;
      for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, v, offset);
        if (lane >= offset) v += n;   // 前 offset 个 lane 没有"上家"
      }
      return v;
    }
    ```

    共 5 步，每步把"已经累加的范围"翻倍（Hillis-Steele 算法）。完整的 block 级和设备级扫描见[前缀和](../kernels/scan.md)。

## 小结

- [x] `__syncthreads()` 必须被 block 内所有线程执行；循环复用共享内存时读后也要同步。
- [x] 原子操作慢在争用；先在 warp、block 内聚合，再做少量全局原子操作。
- [x] 浮点原子加的结果不可复现。
- [x] shuffle 在 warp 内直接交换寄存器；`__shfl_xor_sync` 蝶形归约让所有 lane 拿到结果。
- [x] 不同 block 之间不要互相等待；需要全局同步就拆 kernel，或者用 cooperative launch、线程块集群。
