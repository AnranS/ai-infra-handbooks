# 归约

<p class="lead">归约（把一个数组求和、求最大值）是 CUDA 面试最经典的手写题。它看起来简单，但从最朴素的实现到跑满带宽，要依次解决分支发散、bank 冲突、同步开销、线程利用率、在途访存请求不足等问题，几乎把前面几章的知识点串了一遍。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 共享内存树形归约里，`if (tid % (2*s) == 0)` 的写法有什么问题？
    2. 为什么把"交错寻址"改成"顺序寻址"能消除 bank 冲突？
    3. 为什么让每个线程先串行累加多个元素，再做树形归约，会快得多？
    4. 最后 32 个元素的归约为什么可以不用 `__syncthreads()`？现代写法是什么？
    5. 多个 block 的部分和怎么合并？各有什么优缺点？

## 问题与性能目标

输入 n 个 float，输出它们的和。每个元素只读一次、只做一次加法，**算术强度极低，是纯粹的带宽瓶颈**。所以衡量归约 kernel 的唯一指标是**有效带宽**：`n × 4 字节 / 耗时`，目标是接近显存峰值带宽（实际能达到峰值的 85%-90% 就非常好了）。

整体结构是两步：每个 block 算出自己那部分的和（部分和），再把各 block 的部分和加起来。第二步的数据量很小，下面先专注于第一步。

## 七个版本的演进

完整代码在本节末尾，这里先逐个讲解每个版本改了什么、为什么。

### v0：交错寻址，分支发散

```cuda
for (unsigned s = 1; s < blockDim.x; s *= 2) {
  if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
  __syncthreads();
}
```

第一轮偶数线程工作，第二轮 4 的倍数的线程工作……同一个 warp 内一直有一半以上的线程空闲却要陪跑，**分支发散**严重；取模运算本身也很慢。

### v1：连续的线程做事

```cuda
for (unsigned s = 1; s < blockDim.x; s *= 2) {
  unsigned idx = 2 * s * tid;
  if (idx < blockDim.x) sdata[idx] += sdata[idx + s];
  __syncthreads();
}
```

改成让编号连续的前若干个线程工作，前几轮整个 warp 要么都工作、要么都不工作，发散消失了。但访问模式变成跨步 `2s`：`s = 16` 时，线程 k 访问 `sdata[32k]`，全部落在同一个 bank，产生严重的 **bank 冲突**。

### v2：顺序寻址

```cuda
for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
  if (tid < s) sdata[tid] += sdata[tid + s];
  __syncthreads();
}
```

每一轮把后一半加到前一半上。线程 k 访问 `sdata[k]` 和 `sdata[k + s]`，相邻线程访问相邻地址，**没有 bank 冲突**，也没有发散。这是共享内存树形归约的标准写法。

### v3：加载时先做一次加法

v2 的第一轮就有一半线程闲着。让每个 block 处理 2 倍的数据，每个线程加载两个元素并先相加，再开始树形归约。block 数减半，线程利用率翻倍。

### v4：最后一个 warp 用 shuffle

当 `s ≤ 32` 时，只剩一个 warp 在工作，但每一轮仍然要 `__syncthreads()` 整个 block。用 warp shuffle 完成最后 64 → 1 的归约，省掉最后 5 次 block 同步和 5 次共享内存往返：

```cuda
for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) { ... }
if (tid < 32) {
  float v = sdata[tid] + sdata[tid + 32];
  v = warp_reduce_sum(v);            // 5 次 __shfl_down_sync
  if (tid == 0) out[blockIdx.x] = v;
}
```

!!! note "老教程里的 `volatile` 写法"
    经典资料（Mark Harris 2007 年的 *Optimizing Parallel Reduction in CUDA*）用 `volatile` 共享内存指针、不加同步地展开最后一个 warp，依赖"warp 内线程同步执行"。Volta 引入独立线程调度之后，这种写法不再有保证。现在请使用 `__shfl_down_sync` 或者在每步之间加 `__syncwarp()`。

### v5：每个线程处理多个元素 + 向量化

前面的版本里，每个线程只处理 1 到 2 个元素，就要参与一整套树形归约（`log2(256) = 8` 轮同步），**同步和归约的开销远大于有用的加法**；而且每个线程只有一两个在途的访存请求，按[延迟掩盖](../basics/execution.md#延迟掩盖)一节的分析，跑不满带宽。

v5 做了三件事：

1. **grid-stride loop**：只启动"SM 数 × 几"个 block，每个线程在寄存器里串行累加成百上千个元素，归约开销被摊薄到可以忽略；
2. **`float4` 向量化读取**：每条访存指令读 16 字节，在途数据量是标量版本的 4 倍；
3. **两级 shuffle 归约**：warp 内 shuffle → 各 warp 结果写共享内存 → 第 0 个 warp 再 shuffle。只需要一次 `__syncthreads()`。

这个版本通常能达到峰值带宽的 85% 以上，是实际工程中归约 kernel 的标准结构。softmax、LayerNorm 的"行内归约"用的也是同样的套路。

### v6：cub::DeviceReduce

CUDA 自带的 CUB 库提供了高度优化的设备级归约，工程中直接用它即可。它也是评价你手写版本的基准：

```cuda
size_t temp_bytes = 0;
cub::DeviceReduce::Sum(nullptr, temp_bytes, d_in, d_out, n);   // 第一次调用只查询临时空间大小
cudaMalloc(&d_temp, temp_bytes);
cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);     // 真正执行
```

## 完整代码

```cuda title="reduction.cu"
// reduction.cu —— 归约优化的七个版本，逐个验证正确性并测量有效带宽
// 编译：nvcc -O3 -arch=sm_75 reduction.cu -o reduction
#include "common.cuh"
#include <cub/cub.cuh>

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

// v0：交错寻址 + 取模判断，warp 内严重发散
__global__ void reduce_v0(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[kThreads];
  unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// v1：连续线程工作，没有发散，但跨步访问导致 bank 冲突
__global__ void reduce_v1(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[kThreads];
  unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
    unsigned idx = 2 * stride * tid;
    if (idx < blockDim.x) s[idx] += s[idx + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// v2：顺序寻址，无发散、无 bank 冲突
__global__ void reduce_v2(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[kThreads];
  unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// v3：每个 block 处理 2 * blockDim 个元素，加载时先加一次
__global__ void reduce_v3(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[kThreads];
  unsigned tid = threadIdx.x, i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float v = i < n ? in[i] : 0.f;
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  s[tid] = v;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// v4：v3 + 最后 64 -> 1 用 warp shuffle（要求 blockDim >= 64）
__global__ void reduce_v4(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[kThreads];
  unsigned tid = threadIdx.x, i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float v = i < n ? in[i] : 0.f;
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  s[tid] = v;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid < 32) {
    v = warp_reduce_sum(s[tid] + s[tid + 32]);
    if (tid == 0) out[blockIdx.x] = v;
  }
}

// v5：grid-stride + float4 + 两级 shuffle 归约
__global__ void reduce_v5(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = blockDim.x * gridDim.x;
  float v = 0.f;
  const int n4 = n / 4;
  const float4* in4 = reinterpret_cast<const float4*>(in);
  for (int i = gtid; i < n4; i += nthreads) {
    float4 t = in4[i];
    v += (t.x + t.y) + (t.z + t.w);
  }
  for (int i = n4 * 4 + gtid; i < n; i += nthreads) v += in[i];   // 不足 4 个的尾部

  __shared__ float warp_sums[32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  v = warp_reduce_sum(v);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < blockDim.x / 32 ? warp_sums[lane] : 0.f;
    v = warp_reduce_sum(v);
    if (lane == 0) out[blockIdx.x] = v;
  }
}

struct Version {
  const char* name;
  void (*kernel)(const float*, float*, int);
  int elems_per_block;   // 0 表示用固定的 grid 大小（grid-stride）
};

int main() {
  const int n = (1 << 26) + 3;   // 约 6700 万个元素，故意不是 4 的倍数
  std::vector<float> h(n);
  fill_random(h, 2024, 0.f, 1.f);
  double ref = 0.0;
  for (float v : h) ref += v;

  float *d_in, *d_partial, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partial, ((n + kThreads - 1) / kThreads) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const Version versions[] = {
      {"v0 interleaved+divergent", reduce_v0, kThreads},
      {"v1 interleaved+conflicts", reduce_v1, kThreads},
      {"v2 sequential", reduce_v2, kThreads},
      {"v3 first add on load", reduce_v3, 2 * kThreads},
      {"v4 warp shuffle tail", reduce_v4, 2 * kThreads},
      {"v5 grid-stride+float4", reduce_v5, 0},
  };
  const double bytes = static_cast<double>(n) * sizeof(float);
  bool ok = true;
  std::printf("%-28s %10s %10s  check\n", "version", "time(ms)", "GB/s");
  for (const auto& ver : versions) {
    const int blocks = ver.elems_per_block ? (n + ver.elems_per_block - 1) / ver.elems_per_block : sm_count() * 8;
    ver.kernel<<<blocks, kThreads>>>(d_in, d_partial, n);
    CUDA_CHECK_LAST();
    std::vector<float> partial(blocks);
    CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (float p : partial) sum += p;   // 第二步：部分和很少，这里在 CPU 上合并
    bool good = std::fabs(sum - ref) <= 1e-5 * ref;
    ok &= good;
    float ms = time_ms([&] { ver.kernel<<<blocks, kThreads>>>(d_in, d_partial, n); });
    std::printf("%-28s %10.3f %10.1f  %s\n", ver.name, ms, gbps(bytes, ms), good ? "PASS" : "FAIL");
  }

  // v6：CUB 设备级归约
  size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, d_in, d_out, n));
  void* d_temp;
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
  CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n));
  float cub_sum = 0.f;
  CUDA_CHECK(cudaMemcpy(&cub_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  bool good = std::fabs(cub_sum - ref) <= 1e-4 * ref;
  ok &= good;
  float ms = time_ms([&] { cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n); });
  std::printf("%-28s %10.3f %10.1f  %s\n", "v6 cub::DeviceReduce", ms, gbps(bytes, ms), good ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_partial));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_temp));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
```

运行后把结果填进下表，算出每个版本占你的 GPU 峰值带宽的百分比。这张表本身就是一份很好的学习记录，面试时也可以直接讲：

| 版本 | 主要改动 | 耗时 | 带宽 | 占峰值 |
| --- | --- | --- | --- | --- |
| v0 | 基线 | | | |
| v1 | 消除发散 | | | |
| v2 | 消除 bank 冲突 | | | |
| v3 | 加载时相加 | | | |
| v4 | warp shuffle | | | |
| v5 | 多元素 + 向量化 | | | |
| v6 | CUB | | | |

## 合并各 block 的部分和

| 方法 | 做法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 两次 kernel | 第一个 kernel 写出部分和，第二个 kernel（一个 block）再归约 | 简单、结果确定 | 多一次启动开销 |
| 原子加 | 每个 block 算完后 `atomicAdd` 到结果 | 一个 kernel 完成 | 浮点结果不可复现；需要先清零 |
| 最后一个 block 汇总 | 每个 block 写部分和，`__threadfence()` 后原子地给计数器加一，拿到最后编号的 block 负责汇总 | 一个 kernel、结果确定 | 代码复杂一些 |

CUB 用的是两次 kernel 的做法，并且保证结果确定。

## 面试怎么答

面试官让你"写一个求和的 kernel"时，比较好的节奏是：

1. 先写出 v2 风格的正确版本（顺序寻址），说明为什么不用交错寻址；
2. 主动指出它的问题：每个线程工作量太少、同步太多；
3. 改成 grid-stride + warp shuffle 的 v5 结构，解释 `__shfl_down_sync` 的语义和 mask 参数；
4. 讨论多 block 结果的合并方式和浮点确定性；
5. 说明这是带宽瓶颈的 kernel，衡量标准是有效带宽占峰值的比例，并给出你实测过的数字。

!!! interview "面试怎么答"
    归约是手写 kernel 面试的经典题：它受带宽限制，用有效带宽占峰值的比例衡量。优化顺序：顺序寻址避免发散和 bank 冲突 → 每个线程先在寄存器里串行累加很多元素（并用向量化读取增加在途请求，这是最关键的一步）→ warp 内用 shuffle 归约 → 多个 block 的部分和用第二个 kernel 或原子操作合并。最后说明"warp shuffle → 共享内存 → 再 shuffle"的两级结构是所有按行归约算子（softmax、RMSNorm）的模板，工程上用 CUB 做基准。

## 练习

**1. 求最大值和它的下标（argmax）。** 修改 v5，输出数组中最大值的下标。提示：shuffle 时同时交换值和下标，比较时值相等取较小的下标。

??? success "参考答案"
    ```cuda title="argmax.cu"
    // argmax.cu —— 用 warp shuffle 同时归约值和下标
    // 编译：nvcc -O3 -arch=sm_75 argmax.cu -o argmax
    #include "common.cuh"

    struct Pair {
      float v;
      int i;
    };

    __device__ __forceinline__ Pair better(Pair a, Pair b) {
      return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
    }

    __device__ __forceinline__ Pair warp_argmax(Pair p) {
      for (int offset = 16; offset > 0; offset /= 2) {
        Pair o{__shfl_down_sync(0xffffffff, p.v, offset), __shfl_down_sync(0xffffffff, p.i, offset)};
        p = better(p, o);
      }
      return p;
    }

    __global__ void argmax_kernel(const float* __restrict__ x, int n, float* bv, int* bi) {
      Pair p{-INFINITY, 0x7fffffff};
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        p = better(p, Pair{x[i], i});
      __shared__ Pair ws[32];
      p = warp_argmax(p);
      if (threadIdx.x % 32 == 0) ws[threadIdx.x / 32] = p;
      __syncthreads();
      if (threadIdx.x < 32) {
        p = threadIdx.x < blockDim.x / 32 ? ws[threadIdx.x] : Pair{-INFINITY, 0x7fffffff};
        p = warp_argmax(p);
        if (threadIdx.x == 0) {
          bv[blockIdx.x] = p.v;
          bi[blockIdx.x] = p.i;
        }
      }
    }

    int main() {
      const int n = 10'000'019;
      std::vector<float> h(n);
      fill_random(h, 5);
      h[7'654'321] = 3.f;   // 唯一的最大值
      const int blocks = sm_count() * 4, threads = 256;
      float *dx, *dbv;
      int* dbi;
      CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dbv, blocks * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dbi, blocks * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(dx, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
      argmax_kernel<<<blocks, threads>>>(dx, n, dbv, dbi);
      CUDA_CHECK_LAST();
      std::vector<float> bv(blocks);
      std::vector<int> bi(blocks);
      CUDA_CHECK(cudaMemcpy(bv.data(), dbv, blocks * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(bi.data(), dbi, blocks * sizeof(int), cudaMemcpyDeviceToHost));
      int best = 0;
      for (int b = 1; b < blocks; ++b)
        if (bv[b] > bv[best] || (bv[b] == bv[best] && bi[b] < bi[best])) best = b;
      bool ok = bi[best] == 7'654'321;
      std::printf("argmax = %d (%s)\n", bi[best], ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
    }
    ```

**2. 单 kernel 确定性求和。** 用"最后一个 block 汇总"的方法实现单个 kernel 完成的求和，要求多次运行结果逐位相同。

??? success "参考思路"
    1. 每个 block 算出部分和，写到 `partial[blockIdx.x]`；
    2. 线程 0 执行 `__threadfence()`，确保部分和对其他 block 可见，然后 `unsigned ticket = atomicAdd(&counter, 1)`；
    3. 通过共享内存把"我是不是最后一个"（`ticket == gridDim.x - 1`）广播给整个 block；
    4. 最后一个 block 按**固定顺序**读取所有部分和并归约，写出结果，再把计数器清零以便下次使用。

    因为汇总的顺序固定（按 block 编号），结果与各 block 的完成顺序无关，是确定的。CUDA 官方示例 `threadFenceReduction` 就是这种写法。

## 小结

- [x] 归约是带宽瓶颈，用有效带宽占峰值的比例衡量。
- [x] 顺序寻址避免发散和 bank 冲突；最后一个 warp 用 shuffle。
- [x] 最关键的优化是让每个线程在寄存器里累加大量元素，并用向量化读取增加在途请求。
- [x] warp 内 shuffle → 共享内存 → 再 shuffle 的两级结构，是所有行归约类算子的模板。
- [x] 工程中用 CUB，手写版本以它为基准。
