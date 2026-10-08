# 第一个 CUDA 程序

<p class="lead">这一章写出第一个完整的 CUDA 程序，并把以后每个程序都要用到的"基础设施"准备好：错误检查、计时、结果校验。很多人跳过这些细节，然后在调试时浪费大量时间。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `__global__`、`__device__`、`__host__` 各表示什么？
    2. kernel 启动后，CPU 会等它执行完吗？怎么确认 kernel 执行成功了？
    3. 数组长度不是 block 大小的整数倍时怎么处理？
    4. 什么是 grid-stride loop，它有什么好处？
    5. 怎么准确地测量一个 kernel 的执行时间？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `__global__`：在设备上执行、由主机（或设备）启动的 kernel，返回 void；`__device__`：只能在设备上调用的函数；`__host__`：普通的主机函数，和 `__device__` 一起用可以让同一个函数两边都能编译。
    2. 不会，kernel 启动是异步的。用 `cudaGetLastError()` 检查启动错误（配置不合法等），再用 `cudaDeviceSynchronize()`（或同步的拷贝）等它执行完，并检查执行期间的错误。
    3. grid 的大小向上取整（`(n + block - 1) / block`），kernel 里用 `if (i < n)` 做边界检查，多出来的线程什么都不做。
    4. 每个线程处理下标 `i, i + 总线程数, i + 2 × 总线程数, …`，直到越界。好处：grid 的大小和数据规模解耦，可以按 SM 数量启动固定数量的 block，还能复用线程的初始化开销。
    5. 用 CUDA event 在同一个流里前后打点，`cudaEventElapsedTime` 取间隔；先预热（第一次调用有初始化开销），多次运行取平均或中位数，并说清是哪一种口径。不要用 CPU 计时器直接包住异步的 kernel 启动。要可比还得锁频，并注意两个 event 之间还夹着队列空隙和启动延迟。

## 程序的基本结构

一个 CUDA 程序由两部分代码组成：在 CPU 上运行的**主机代码（host）**和在 GPU 上运行的**设备代码（device）**。函数前的修饰符决定它在哪里运行、能从哪里调用：

| 修饰符 | 在哪里执行 | 从哪里调用 | 说明 |
| --- | --- | --- | --- |
| `__global__` | GPU | CPU（或 GPU，动态并行） | kernel 函数，返回值必须是 `void` |
| `__device__` | GPU | GPU | 设备端的普通函数，通常会被内联 |
| `__host__` | CPU | CPU | 默认值，可以省略 |
| `__host__ __device__` | 两边都编译一份 | 两边都可调用 | 适合写 CPU 和 GPU 通用的小函数 |

典型的执行流程只有五步：

1. 在 GPU 上分配内存（`cudaMalloc`）；
2. 把输入数据从 CPU 拷贝到 GPU（`cudaMemcpy`）；
3. 启动 kernel；
4. 把结果拷回 CPU；
5. 释放 GPU 内存（`cudaFree`）。

## 公共工具：common.cuh

本手册后面的所有示例都会包含这个头文件。它提供错误检查宏、计时器、随机初始化和结果校验：

```cuda title="common.cuh"
// common.cuh —— 本手册所有示例共用的小工具
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

// 检查 CUDA API 的返回值，出错时打印位置并退出
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                     \
                   cudaGetErrorName(err_), __FILE__, __LINE__,                 \
                   cudaGetErrorString(err_));                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// kernel 启动之后调用：立即报告启动配置错误
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// 用 CUDA event 计时，单位毫秒
struct GpuTimer {
  cudaEvent_t start_, stop_;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }
  float stop(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(stop_, s));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

// 先预热 warmup 次，再运行 iters 次取平均耗时（毫秒）
template <typename F>
float time_ms(F&& fn, int iters = 20, int warmup = 3) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaGetLastError());
  GpuTimer t;
  t.start();
  for (int i = 0; i < iters; ++i) fn();
  float ms = t.stop() / iters;
  CUDA_CHECK(cudaGetLastError());
  return ms;
}

// 逐次采样：返回（最小值，中位数），单位毫秒。和 time_ms 的"整批平均"是两种口径，不能混着比
template <typename F>
std::pair<float, float> time_stats(F&& fn, int iters = 50, int warmup = 5) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> ms(iters);
  GpuTimer t;
  for (int i = 0; i < iters; ++i) {
    t.start();
    fn();
    ms[i] = t.stop();                               // 每次都同步一下，拿到的是单次样本
  }
  CUDA_CHECK(cudaGetLastError());
  std::sort(ms.begin(), ms.end());
  return {ms.front(), ms[iters / 2]};
}

inline void fill_random(std::vector<float>& v, unsigned seed = 42, float lo = -1.f, float hi = 1.f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& x : v) x = dist(gen);
}

// 与 CPU 参考结果比较：|got - ref| <= atol + rtol * |ref| 视为正确
inline bool check_close(const float* got, const float* ref, size_t n,
                        float rtol = 1e-4f, float atol = 1e-5f) {
  size_t bad = 0;
  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
    max_err = std::max(max_err, err);
    if (!(err <= atol + rtol * std::fabs(ref[i]))) {  // 取反写法能同时捕获 NaN
      if (bad < 5) std::fprintf(stderr, "  mismatch at %zu: got %g, expected %g\n", i, got[i], ref[i]);
      ++bad;
    }
  }
  std::printf("%s  max_abs_err=%.3e  mismatches=%zu/%zu\n", bad ? "FAIL" : "PASS", max_err, bad, n);
  return bad == 0;
}

// 当前 GPU 的计算能力低于要求时打印 SKIP 并正常退出，方便批量运行示例
inline void require_sm(int major, int minor = 0) {
  int dev = 0;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  if (p.major * 10 + p.minor < major * 10 + minor) {
    std::printf("SKIP: needs sm_%d%d, this GPU (%s) is sm_%d%d\n", major, minor, p.name, p.major, p.minor);
    std::exit(0);
  }
}

inline int sm_count() {
  int dev = 0, n = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  return n;
}

// 带宽（GB/s）与算力（TFLOPS）换算
inline double gbps(double bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }
inline double tflops(double flops, float ms) { return flops / (ms * 1e-3) / 1e12; }
```

## 向量加法

```cuda title="vector_add.cu"
// vector_add.cu —— 第一个完整的 CUDA 程序
// 编译：nvcc -O3 -arch=sm_75 vector_add.cu -o vector_add
#include "common.cuh"

// 每个线程负责一个元素
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程编号
  if (i < n) {                                    // 最后一个 block 可能有多余的线程
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int n = 1 << 24;  // 约 1600 万个元素
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  // 1. 准备主机数据和 CPU 参考结果
  std::vector<float> h_a(n), h_b(n), h_c(n), ref(n);
  fill_random(h_a, 1);
  fill_random(h_b, 2);
  for (int i = 0; i < n; ++i) ref[i] = h_a[i] + h_b[i];

  // 2. 分配显存并拷贝输入
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  // 3. 启动 kernel：向上取整，保证覆盖所有元素
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
  CUDA_CHECK_LAST();                     // 启动配置是否有错
  CUDA_CHECK(cudaDeviceSynchronize());   // 等待执行完，并报告执行期间的错误

  // 4. 拷回结果并校验
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  check_close(h_c.data(), ref.data(), n);

  // 5. 计时：读两个数组、写一个数组
  float ms = time_ms([&] { vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n); });
  std::printf("vector_add: %.3f ms, %.1f GB/s\n", ms, gbps(3.0 * bytes, ms));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}
```

几个要点：

- **`<<<blocks, threads>>>`** 是 kernel 的**执行配置**。block 大小一般取 128、256、512，这里 256 是稳妥的默认值。
- **边界检查 `if (i < n)`** 必不可少。block 数是向上取整的，最后一个 block 里可能有线程超出数组范围。
- **下标溢出**：`blockIdx.x * blockDim.x` 是 `unsigned int` 乘法，元素超过 2^31 个时，下标要用 64 位：`size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;`。大模型场景下的张量经常超过这个规模。
- 这个 kernel 是纯粹的**访存瓶颈**：每个元素 12 字节，只做一次加法。衡量它的唯一标准是实测带宽占显存峰值带宽的比例。

拨一拨元素个数和 blockDim，看每个线程算哪个元素、末尾多出的线程怎么处理：

<div class="aig-widget" data-widget="grid-index"></div>

## kernel 启动是异步的

kernel 启动后 CPU **立即返回**，不会等 GPU 执行完。GPU 按提交顺序依次执行同一个流里的操作。以下情况 CPU 会等待：

- `cudaMemcpy`（默认流上的同步拷贝）会等之前的 kernel 执行完再开始拷贝；
- `cudaDeviceSynchronize()` 等待设备上所有工作完成；
- `cudaEventSynchronize(e)`、`cudaStreamSynchronize(s)` 等待特定的事件或流。

异步带来了两个常见的坑：

**计时错误**。用 CPU 计时器包住 kernel 启动，测到的只是"启动"的时间：

```cuda
auto t0 = std::chrono::steady_clock::now();
kernel<<<grid, block>>>(...);
auto t1 = std::chrono::steady_clock::now();   // 错误：kernel 可能还没开始执行
```

正确的做法是用 CUDA event 计时（`GpuTimer`），或者在停止计时前 `cudaDeviceSynchronize()`。另外，第一次启动 kernel 包含模块加载、JIT 等一次性开销，所以要**先预热**再计时，并**多次运行取平均**。`time_ms` 就是这样做的。

**错误报告延迟**。kernel 执行中的错误（比如越界访问）不会在启动那一行报告，而是在之后的某个同步点才被发现。

## 错误处理

CUDA 的错误分两类：

| 类型 | 例子 | 何时能发现 |
| --- | --- | --- |
| 启动错误 | block 大小超过 1024、共享内存申请过多、没有适合当前 GPU 的 kernel 镜像 | 启动后立即调用 `cudaGetLastError()` |
| 执行错误 | 非法地址访问、未对齐访问、kernel 里的 `assert` 失败 | 下一个同步调用的返回值 |

执行错误是**粘滞的（sticky）**：一旦发生，这个进程里后续所有 CUDA 调用都会返回错误，只能重启进程。所以看到一个奇怪的错误时，真正的原因往往在更早的某个 kernel 里。调试时可以设置环境变量 `CUDA_LAUNCH_BLOCKING=1`，让每次 kernel 启动都同步执行，错误就会在出问题的那一行报告。

**每个 CUDA API 调用都要检查返回值。** 这是写 CUDA 代码最重要的习惯，也是面试官看代码时会注意的细节。

### compute-sanitizer：越界和竞争检查

CUDA Toolkit 自带的 `compute-sanitizer` 相当于 GPU 版的 AddressSanitizer：

```bash
compute-sanitizer ./vector_add                      # 默认 memcheck：越界、未对齐、非法地址
compute-sanitizer --tool racecheck ./app            # 共享内存的数据竞争
compute-sanitizer --tool initcheck ./app            # 读取未初始化的显存
compute-sanitizer --tool synccheck ./app            # 同步原语的错误用法
```

程序会慢几十倍，但能精确定位到出错的线程和源代码行（编译时加 `-lineinfo`）。**写完一个新 kernel，先用 memcheck 跑一遍小规模数据**，这能省下大量调试时间。

## 让测出来的数字可信

改完一版 kernel，时间变短了——怎么确定它真的更快？下面几条是比优化技巧更早需要的东西。

**先说清口径**。"一次向量加法耗时"包不包括分配显存、拷数据、校验结果？两个范围都有用，但比较之前必须声明是哪一个。本手册统一的口径是：**输入输出都已经在显存上，预热之后重复执行 kernel，用 CUDA event 量这段时间**；分配、初始化、`check_close` 都在计时区间外。

**event 之间不只有 kernel**。`cudaEventElapsedTime` 给的是两个标记在 GPU 时间线上的时间戳之差。GPU 按提交顺序执行：如果 start 标记执行完时队列空了、CPU 还没把 kernel 提交上来，这段空闲同样算在里面。所以 **kernel 越短，测量值里混进的启动延迟占比越大**；想知道 kernel 自身的起止时间，要用 [Nsight](../tools/profiling.md)。这也是"我 event 测出来比 ncu 报的大"的常见原因。

**两种统计口径**。`common.cuh` 里有两个计时函数，它们测的不是同一件事：

```cuda
float ms = time_ms(run, 20, 3);                     // 两个 event 包住 20 次调用，总时间 / 20：整批平均
auto [min_ms, med_ms] = time_stats(run, 50, 5);     // 逐次打点，拿到 50 个样本，取最小值与中位数
```

- **整批平均**少了 19 次插 event 和同步的开销，但从一个总时间里**恢复不出**单次的分布；
- **逐次采样**能看到波动：**中位数**不受个别特别慢的样本影响，**最小值**是"最好情况"的参考，但别拿它当"每次都能达到"的速度；
- 三个数（平均、中位数、最小值）谁也不等于谁。本手册各章的数字都用 `time_ms`，也就是整批平均，所以可以互相比较。

**把频率锁住**。GPU 的 SM 和显存频率会随温度、功耗墙上下浮动，不锁频的话同一个程序前后两次差几个百分点很正常：

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS | head -20      # 看这张卡支持哪些频率
sudo nvidia-smi -lgc 1500,1500                    # 把 SM 频率锁在 1500 MHz
sudo nvidia-smi --lock-memory-clocks=9501,9501    # 显存频率也锁住（可选）
# ... 跑基准测试 ...
sudo nvidia-smi -rgc && sudo nvidia-smi -rmc      # 测完恢复
```

Nsight Compute 默认会替你做类似的事（`--clock-control base` 锁基础频率、`--cache-control all` 每次重放前清缓存），代价是绝对耗时和正常运行时不同。

**有效带宽不是显存总线上的实测流量**。我们算的 `gbps(bytes, ms)` 用的是**算法要求的字节数**（读了几个数组、写了几个数组）除以时间。数据量小的时候，这些字节可能大部分命中 L2，根本没经过显存总线，于是算出来的"有效带宽"会高得离谱，甚至超过这张卡的理论峰值。所以：

- 测显存带宽要让**工作集远大于 L2**（几十 MB 以上），并报告数组大小；
- 数字异常高时先怀疑缓存，再怀疑自己写得好；真实的 DRAM 流量要用 Nsight 的 `dram__bytes` 之类的计数器看。

**单位别混**。`MiB` 是 $2^{20}$ 字节，`GB` 是 $10^9$ 字节。先算出准确的字节数，再统一换算成 GB/s——直接拿 MiB 去除以毫秒会差 4.9%。

**校验放在计时区间外**，而且要真的校验：`check_close` 会把 NaN 也判成失败（取反的写法），不然一个全是 NaN 的 kernel 可能跑得飞快。

把上面几条合成一个可以直接跑的基准程序。它打印运行环境，然后做三个实验：同一个 kernel 的三种统计口径、复制不同大小数组时的有效带宽、以及一个什么都不做的 kernel 的耗时下限。

```cuda title="bench_timing.cu"
// bench_timing.cu —— 这一节的三个实验：口径差异、工作集与缓存、短 kernel 的启动延迟
// 编译：nvcc -O3 -arch=sm_75 bench_timing.cu -o bench_timing
// 在自己的卡上建议改成 -arch=native，省掉第一次运行的 JIT
#include "common.cuh"

__global__ void vector_add(const float* a, const float* b, float* c, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void copy_kernel(const float* src, float* dst, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

__global__ void empty_kernel() {}

static void report_env() {
  int dev = 0, rt = 0, drv = 0, clk_khz = 0, mem_khz = 0, bus = 0, l2 = 0;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
  double peak = 2.0 * mem_khz * 1e3 * bus / 8.0 / 1e9;   // DDR 类显存每个时钟传两次
  std::printf("GPU: %s (sm_%d%d, %d SM, L2 %.1f MiB)\n", p.name, p.major, p.minor, p.multiProcessorCount,
              l2 / 1048576.0);
  std::printf("SM 时钟 %.0f MHz, 显存 %.0f MHz x %d bit -> 理论峰值带宽 %.0f GB/s\n",
              clk_khz / 1000.0, mem_khz / 1000.0, bus, peak);
  std::printf("CUDA runtime %d.%d, driver %d.%d\n\n", rt / 1000, rt % 1000 / 10, drv / 1000, drv % 1000 / 10);
}

static double peak_gbps() {
  int dev = 0, mem_khz = 0, bus = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));
  return 2.0 * mem_khz * 1e3 * bus / 8.0 / 1e9;
}

// 实验一：同一个 kernel，三种统计口径给出三个数
static void experiment_scope() {
  const long long n = 4096LL * 4096;                      // 1600 万个元素，三个数组共 192 MiB
  const size_t bytes = n * sizeof(float);
  float *a, *b, *c;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMalloc(&c, bytes));
  std::vector<float> ha(n, 1.f), hb(n, 2.f), hc(n);
  CUDA_CHECK(cudaMemcpy(a, ha.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b, hb.data(), bytes, cudaMemcpyHostToDevice));
  int block = 256;
  long long grid = (n + block - 1) / block;
  auto run = [&] { vector_add<<<(unsigned)grid, block>>>(a, b, c, n); };

  float batch = time_ms(run, 200, 20);                    // 两个 event 包住 200 次，总时间 / 200
  auto [lo, med] = time_stats(run, 200, 20);              // 逐次打点的 200 个样本
  double moved = 3.0 * n * sizeof(float);                 // 算法字节数：读两个数组、写一个
  std::printf("--- 实验一：同一个向量加法，三种口径 ---\n");
  std::printf("%-18s %10s %12s\n", "统计量", "毫秒", "GB/s");
  std::printf("%-18s %10.6f %12.1f\n", "整批平均", batch, gbps(moved, batch));
  std::printf("%-18s %10.6f %12.1f\n", "逐次中位数", med, gbps(moved, med));
  std::printf("%-18s %10.6f %12.1f\n", "逐次最小值", lo, gbps(moved, lo));

  CUDA_CHECK(cudaMemcpy(hc.data(), c, bytes, cudaMemcpyDeviceToHost));
  std::vector<float> ref(n, 3.f);                         // 校验在计时区间之外
  std::printf("校验：");
  check_close(hc.data(), ref.data(), n);
  std::printf("\n");
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
}

// 实验二：复制不同大小的数组，看"有效带宽"怎样随工作集变化
static void experiment_working_set() {
  const double peak = peak_gbps();
  std::printf("--- 实验二：工作集大小与有效带宽（复制，一读一写）---\n");
  std::printf("%10s %12s %10s %10s %8s %s\n", "单数组 MiB", "读写 MiB", "中位 ms", "GB/s", "占峰值", "校验");
  for (int mib : {1, 4, 16, 64, 256}) {
    long long n = (long long)mib * 1048576 / sizeof(float);
    size_t bytes = n * sizeof(float);
    float *src, *dst;
    CUDA_CHECK(cudaMalloc(&src, bytes));
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    std::vector<float> h(n);
    fill_random(h, 7);
    CUDA_CHECK(cudaMemcpy(src, h.data(), bytes, cudaMemcpyHostToDevice));
    int block = 256;
    long long grid = (n + block - 1) / block;
    auto run = [&] { copy_kernel<<<(unsigned)grid, block>>>(src, dst, n); };
    auto [lo, med] = time_stats(run, 200, 20);
    (void)lo;
    double moved = 2.0 * bytes;
    std::vector<float> back(n);
    CUDA_CHECK(cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) ok = back[i] == h[i];
    std::printf("%10d %12.0f %10.6f %10.1f %7.0f%% %s\n", mib, 2.0 * mib, med, gbps(moved, med),
                100.0 * gbps(moved, med) / peak, ok ? "PASS" : "FAIL");
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
  }
  std::printf("\n");
}

// 实验三：kernel 越短，测量值里混进的启动延迟占比越大
static void experiment_overhead() {
  auto run = [] { empty_kernel<<<1, 1>>>(); };
  float batch = time_ms(run, 1000, 50);
  auto [lo, med] = time_stats(run, 1000, 50);
  std::printf("--- 实验三：一个什么都不做的 kernel ---\n");
  std::printf("整批平均 %.6f ms，逐次中位数 %.6f ms，逐次最小值 %.6f ms\n", batch, med, lo);
  std::printf("逐次测量的这个下限就是「排队 + 提交」的开销；kernel 自身耗时接近它时，event 的读数基本在量开销\n\n");
}

int main() {
  report_env();
  experiment_scope();
  experiment_working_set();
  experiment_overhead();
  return 0;
}
```

这个程序不在本机运行（手册的 CUDA 代码只做编译检查），把它编译出来在自己的卡上跑一遍，三个表就是上面几条结论的证据。跑之前先锁频，跑完记下 GPU 型号、驱动、CUDA 版本和日期。

**记下来才能复查**。一次基准测试值得记的五件事：机器与日期（GPU 型号、驱动、CUDA 版本）、完整命令、输入规模与校验方式、计时口径与统计量、以及**哪些是观察、哪些是还没验证的解释**。"小数组带宽更高"是观察，"因为命中了 L2"是待验证的解释——下一步该去 profiler 里找证据，而不是直接写进结论。

## grid-stride loop

上面的写法要求线程数至少等于元素数。另一种常用写法是**网格跨步循环**：启动固定数量的线程，每个线程以"整个 grid 的线程总数"为步长循环处理多个元素：

```cuda title="grid_stride.cu"
// grid_stride.cu —— 网格跨步循环实现 SAXPY: y = a * x + y
// 编译：nvcc -O3 -arch=sm_75 grid_stride.cu -o grid_stride
#include "common.cuh"

__global__ void saxpy(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < static_cast<size_t>(n);
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  const int n = 1 << 25;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> h_x(n), h_y(n), ref(n);
  fill_random(h_x, 1);
  fill_random(h_y, 2);
  const float a = 2.5f;
  for (int i = 0; i < n; ++i) ref[i] = a * h_x[i] + h_y[i];

  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

  // block 数只和 GPU 规模有关，和数据量无关：每个 SM 放若干个 block
  const int threads = 256;
  const int blocks = sm_count() * 8;
  saxpy<<<blocks, threads>>>(n, a, d_x, d_y);
  CUDA_CHECK_LAST();
  std::vector<float> h_out(n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  check_close(h_out.data(), ref.data(), n);

  // 计时会反复更新 y，只关心速度
  float ms = time_ms([&] { saxpy<<<blocks, threads>>>(n, a, d_x, d_y); });
  std::printf("saxpy (grid-stride, %d blocks): %.3f ms, %.1f GB/s\n", blocks, ms, gbps(3.0 * bytes, ms));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return 0;
}
```

grid-stride loop 的好处：

- **任意规模的数据**都能处理，不受 grid 大小限制；
- block 数可以按 GPU 规模选择，**线程数少了，线程的启动和收尾开销也少了**，在每个元素的工作量很小时往往更快；
- 同一个 kernel 用 `<<<1, 1>>>` 启动就变成串行执行，方便调试。

`__restrict__` 告诉编译器指针之间没有重叠（别名），编译器可以更激进地优化，比如把只读数据走只读缓存。确定指针不重叠时应该加上。

## 查询设备信息

写 kernel 之前，先搞清楚手上的 GPU 有哪些资源：

```cuda title="device_query.cu"
// device_query.cu —— 打印 GPU 的关键参数
// 编译：nvcc -O3 -arch=sm_75 device_query.cu -o device_query
#include "common.cuh"

int main() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int dev = 0; dev < count; ++dev) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int mem_clock_khz = 0, bus_width = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, dev));
    // DDR 类显存每个时钟传两次数据
    double peak_bw = 2.0 * mem_clock_khz * 1e3 * (bus_width / 8) / 1e9;

    std::printf("GPU %d: %s (sm_%d%d)\n", dev, p.name, p.major, p.minor);
    std::printf("  SMs                         : %d\n", p.multiProcessorCount);
    std::printf("  global memory               : %.1f GB\n", p.totalGlobalMem / 1e9);
    std::printf("  theoretical bandwidth       : %.0f GB/s\n", peak_bw);
    std::printf("  L2 cache                    : %d KB\n", p.l2CacheSize / 1024);
    std::printf("  shared memory per SM        : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    std::printf("  shared memory per block     : %zu KB (opt-in max %zu KB)\n",
                p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
    std::printf("  registers per SM / block    : %d / %d\n", p.regsPerMultiprocessor, p.regsPerBlock);
    std::printf("  max threads per SM / block  : %d / %d\n", p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock);
    std::printf("  max blocks per SM           : %d\n", p.maxBlocksPerMultiProcessor);
    std::printf("  warp size                   : %d\n", p.warpSize);
  }
  return 0;
}
```

`sharedMemPerBlock` 是默认上限（48 KB），`sharedMemPerBlockOptin` 是显式申请后能用的上限，后面写 GEMM 时会用到。

!!! interview "怎么讲清楚"
    这一章常以"写一个向量加法并说明细节"的形式出现：`__global__` 在设备上执行、由主机启动；kernel 启动是异步的，要用 `cudaGetLastError` 查启动错误、用同步或 event 等结果；长度不是 block 大小的整数倍时做边界检查，下标注意 64 位溢出；grid-stride loop 让 kernel 与数据规模解耦。计时用 CUDA event、先预热、多次取平均；新写的 kernel 先过一遍 `compute-sanitizer`。能主动说出这些"工程习惯"，比只会写出正确的 kernel 更加分。

## 练习

**1. 二维矩阵加法。** 写一个 kernel 计算 `C = A + B`，矩阵大小 `rows × cols`，按行主序存储。使用二维的 block（`dim3 block(32, 8)`）和二维的 grid。想一想：`threadIdx.x` 应该对应行还是列？为什么？

??? success "参考答案"
    ```cuda title="matrix_add.cu"
    // matrix_add.cu —— 二维线程组织
    // 编译：nvcc -O3 -arch=sm_75 matrix_add.cu -o matrix_add
    #include "common.cuh"

    __global__ void matrix_add(const float* A, const float* B, float* C, int rows, int cols) {
      int col = blockIdx.x * blockDim.x + threadIdx.x;  // x 对应列：相邻线程访问相邻地址
      int row = blockIdx.y * blockDim.y + threadIdx.y;
      if (row < rows && col < cols) {
        size_t idx = static_cast<size_t>(row) * cols + col;
        C[idx] = A[idx] + B[idx];
      }
    }

    int main() {
      const int rows = 3000, cols = 5000;  // 故意取不能整除的大小
      const size_t n = static_cast<size_t>(rows) * cols, bytes = n * sizeof(float);
      std::vector<float> hA(n), hB(n), hC(n), ref(n);
      fill_random(hA, 1);
      fill_random(hB, 2);
      for (size_t i = 0; i < n; ++i) ref[i] = hA[i] + hB[i];

      float *A, *B, *C;
      CUDA_CHECK(cudaMalloc(&A, bytes));
      CUDA_CHECK(cudaMalloc(&B, bytes));
      CUDA_CHECK(cudaMalloc(&C, bytes));
      CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

      dim3 block(32, 8);
      dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
      matrix_add<<<grid, block>>>(A, B, C, rows, cols);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
      check_close(hC.data(), ref.data(), n);

      float ms = time_ms([&] { matrix_add<<<grid, block>>>(A, B, C, rows, cols); });
      std::printf("matrix_add: %.3f ms, %.1f GB/s\n", ms, gbps(3.0 * bytes, ms));
      CUDA_CHECK(cudaFree(A));
      CUDA_CHECK(cudaFree(B));
      CUDA_CHECK(cudaFree(C));
      return 0;
    }
    ```

    `threadIdx.x` 应该对应**列**（最内层、连续存储的维度）。同一个 warp 的 32 个线程 `threadIdx.x` 连续，访问同一行里连续的 32 个元素，内存访问能合并。如果反过来让 x 对应行，一个 warp 的线程会访问 32 个不同的行，每次访存都跨越一整行，带宽会下降好几倍。下一章会详细解释原因。

**2. 找 bug。** 下面的程序在 `n = 1000` 时结果正确，在某些 GPU 上 `n` 很大时结果随机出错或者程序报错。至少找出 3 个问题。

```cuda
__global__ void scale(float* x, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  x[i] = x[i] * s;
}

int main() {
  int n = 3'000'000'000 / 4;
  float* d;
  cudaMalloc(&d, n * sizeof(float));
  scale<<<n / 1024, 2048>>>(d, 2.0f, n);
  cudaDeviceSynchronize();
}
```

??? success "参考答案"
    1. **没有边界检查**：`n / 1024` 向下取整，而且没有 `if (i < n)`，要么漏处理元素，要么越界。
    2. **block 大小 2048 超过上限 1024**：kernel 根本没有启动，但因为没检查错误，程序"静默成功"。
    3. **没有检查任何 CUDA 调用的返回值**：`cudaMalloc` 失败（比如显存不够）也不会知道。
    4. **显存没有初始化也没有释放**：`cudaMalloc` 分配的内存内容是未定义的。
    5. 这里 `n` 约 7.5 亿，还在 `int` 范围内；但如果数据再大 3 倍，`int i` 和 `n * sizeof(float)` 之前的 `n` 都会溢出，大规模数据应该统一使用 `size_t` / `int64_t`。

**3. 用 compute-sanitizer 抓越界。** 把 `vector_add` 里的 `if (i < n)` 删掉，把 `n` 改成一个不能被 256 整除的数，分别直接运行和用 `compute-sanitizer` 运行，对比输出。

??? success "参考答案"
    直接运行时，越界写入的地址很可能落在 `cudaMalloc` 分配的内存块的对齐填充里，程序往往**看起来正常**。这正是越界 bug 危险的地方。`compute-sanitizer` 会报告类似 `Invalid __global__ write of size 4 bytes`，并指出线程编号和地址。编译时加 `-lineinfo` 还能看到出错的源代码行。

## 小结

- [x] 每个 CUDA 调用都检查返回值；kernel 启动后用 `cudaGetLastError` 检查启动错误。
- [x] kernel 启动是异步的；计时用 CUDA event，先预热，多次取平均。
- [x] 比较之前先说清口径（计时范围 + 统计量）；两个 event 之间还夹着队列空隙和启动延迟，kernel 越短影响越大；要可比就锁频。
- [x] 有效带宽是「算法字节数 / 时间」，工作集小的时候可能只测到了 L2；校验放在计时区间外。
- [x] 索引计算注意边界检查和 64 位溢出。
- [x] 新写的 kernel 先用 `compute-sanitizer` 跑一遍。
- [x] grid-stride loop 让 kernel 与数据规模解耦，是很多库的标准写法。
