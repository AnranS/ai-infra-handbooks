# Your first CUDA program

<p class="lead">This chapter writes the first complete CUDA program and sets up the infrastructure every later program needs: error checking, timing and result verification. Many people skip these details and then lose a great deal of time debugging.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What do `__global__`, `__device__` and `__host__` mean?
    2. Does the CPU wait for a kernel after launching it? How do you confirm a kernel succeeded?
    3. What do you do when the array length is not a multiple of the block size?
    4. What is a grid-stride loop and what does it buy?
    5. How do you measure a kernel's execution time accurately?

??? success "Answers (try it yourself first, then expand)"
    1. `__global__`: a kernel that runs on the device and is launched from the host (or the device), returning void; `__device__`: a function callable only from the device; `__host__`: an ordinary host function, which combined with `__device__` lets one function compile for both sides.
    2. It does not; a kernel launch is asynchronous. Check launch errors (an invalid configuration and so on) with `cudaGetLastError()`, then wait with `cudaDeviceSynchronize()` (or a synchronous copy) and check for errors during execution.
    3. Round the grid up (`(n + block - 1) / block`) and guard with `if (i < n)` inside the kernel so the extra threads do nothing.
    4. Each thread handles indices `i, i + total threads, i + 2 × total threads, …` until it runs out. The benefits: the grid size is decoupled from the data size, you can launch a fixed number of blocks based on the SM count, and the threads' setup cost is reused.
    5. Record CUDA events before and after on the same stream and take the interval with `cudaEventElapsedTime`; warm up first (the first call carries initialization overhead) and take a mean or median over several runs — and say which one. Never wrap an asynchronous launch in a CPU timer. For results that can be compared, lock the clocks as well, and remember that queue gaps and launch latency also fall between the two events.

## The shape of a program {#程序的基本结构}

A CUDA program is two kinds of code: **host** code running on the CPU and **device** code running on the GPU. A qualifier before a function decides where it runs and from where it can be called:

| Qualifier | Runs on | Called from | Notes |
| --- | --- | --- | --- |
| `__global__` | GPU | CPU (or GPU, with dynamic parallelism) | a kernel; the return type must be `void` |
| `__device__` | GPU | GPU | an ordinary device-side function, usually inlined |
| `__host__` | CPU | CPU | the default, and can be omitted |
| `__host__ __device__` | compiled for both | callable from both | good for small functions shared by CPU and GPU |

The typical flow is only five steps:

1. allocate memory on the GPU (`cudaMalloc`);
2. copy the input from CPU to GPU (`cudaMemcpy`);
3. launch the kernel;
4. copy the result back;
5. free the GPU memory (`cudaFree`).

## Shared utilities: common.cuh {#公共工具commoncuh}

Every later example in the handbook includes this header. It provides the error-checking macros, a timer, random initialization and result verification:

```cuda title="common.cuh"
// common.cuh - the small utilities every example in this handbook shares
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

// check a CUDA API return value; on failure print where and exit
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

// call right after a launch: reports a bad launch configuration immediately
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// time with CUDA events, in milliseconds
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

// warm up `warmup` times, then average over `iters` runs (milliseconds)
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

// Per-call samples: returns (minimum, median) in milliseconds. This and time_ms's "batch average" are two different conventions; never compare one against the other
template <typename F>
std::pair<float, float> time_stats(F&& fn, int iters = 50, int warmup = 5) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> ms(iters);
  GpuTimer t;
  for (int i = 0; i < iters; ++i) {
    t.start();
    fn();
    ms[i] = t.stop();                               // synchronising each round is what makes these single-call samples
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

// compare against the CPU reference: correct when |got - ref| <= atol + rtol * |ref|
inline bool check_close(const float* got, const float* ref, size_t n,
                        float rtol = 1e-4f, float atol = 1e-5f) {
  size_t bad = 0;
  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
    max_err = std::max(max_err, err);
    if (!(err <= atol + rtol * std::fabs(ref[i]))) {  // writing it negated catches NaN as well
      if (bad < 5) std::fprintf(stderr, "  mismatch at %zu: got %g, expected %g\n", i, got[i], ref[i]);
      ++bad;
    }
  }
  std::printf("%s  max_abs_err=%.3e  mismatches=%zu/%zu\n", bad ? "FAIL" : "PASS", max_err, bad, n);
  return bad == 0;
}

// print SKIP and exit cleanly when this GPU's compute capability is too low, so the examples can be run in bulk
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

// converting to bandwidth (GB/s) and throughput (TFLOPS)
inline double gbps(double bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }
inline double tflops(double flops, float ms) { return flops / (ms * 1e-3) / 1e12; }
```

## Vector addition {#向量加法}

```cuda title="vector_add.cu"
// vector_add.cu - the first complete CUDA program
// build: nvcc -O3 -arch=sm_75 vector_add.cu -o vector_add
#include "common.cuh"

// one element per thread
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // the global thread index
  if (i < n) {                                    // the last block may hold extra threads
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int n = 1 << 24;  // about 16 million elements
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  // 1. prepare the host data and the CPU reference
  std::vector<float> h_a(n), h_b(n), h_c(n), ref(n);
  fill_random(h_a, 1);
  fill_random(h_b, 2);
  for (int i = 0; i < n; ++i) ref[i] = h_a[i] + h_b[i];

  // 2. allocate device memory and copy the input over
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  // 3. launch the kernel, rounding up so every element is covered
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
  CUDA_CHECK_LAST();                     // was the launch configuration wrong?
  CUDA_CHECK(cudaDeviceSynchronize());   // wait for it to finish and report any error during execution

  // 4. copy the result back and verify it
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  check_close(h_c.data(), ref.data(), n);

  // 5. timing: two arrays read, one written
  float ms = time_ms([&] { vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n); });
  std::printf("vector_add: %.3f ms, %.1f GB/s\n", ms, gbps(3.0 * bytes, ms));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}
```

A few points:

- **`<<<blocks, threads>>>`** is the kernel's **execution configuration**. Block sizes are usually 128, 256 or 512, and 256 is a safe default.
- **The bounds check `if (i < n)`** is essential. The block count is rounded up, so the last block may hold threads past the end of the array.
- **Index overflow**: `blockIdx.x * blockDim.x` is `unsigned int` arithmetic, so beyond 2^31 elements the index must be 64-bit: `size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;`. Tensors in LLM work routinely exceed that.
- This kernel is purely **memory-bound**: 12 bytes per element for one addition. The only measure of it is what fraction of peak memory bandwidth it achieves.

Move the element count and blockDim around to see which element each thread computes and what happens to the extra threads at the end:

<div class="aig-widget" data-widget="grid-index"></div>

## A kernel launch is asynchronous {#kernel-启动是异步的}

The CPU **returns immediately** after a launch and does not wait for the GPU. The GPU runs the operations of one stream in submission order. The CPU waits in these cases:

- `cudaMemcpy` (a synchronous copy on the default stream) waits for earlier kernels before copying;
- `cudaDeviceSynchronize()` waits for all work on the device;
- `cudaEventSynchronize(e)` and `cudaStreamSynchronize(s)` wait for a particular event or stream.

Asynchrony brings two common traps:

**Timing mistakes.** Wrapping a launch in a CPU timer measures only the launch:

```cuda
auto t0 = std::chrono::steady_clock::now();
kernel<<<grid, block>>>(...);
auto t1 = std::chrono::steady_clock::now();   // wrong: the kernel may not even have started
```

The right way is to time with CUDA events (`GpuTimer`), or to call `cudaDeviceSynchronize()` before stopping the timer. Note also that the first launch includes one-off costs such as module loading and JIT, so **warm up first** and **average over several runs**. That is what `time_ms` does.

**Delayed error reporting.** An error during execution (an out-of-bounds access, say) is not reported at the launch line but discovered at some later synchronization point.

## Error handling {#错误处理}

CUDA errors come in two kinds:

| Kind | Examples | When you can see it |
| --- | --- | --- |
| launch errors | block size above 1024, too much shared memory requested, no kernel image for this GPU | `cudaGetLastError()` right after the launch |
| execution errors | an illegal address, a misaligned access, a failed `assert` in the kernel | the return value of the next synchronizing call |

Execution errors are **sticky**: once one happens, every later CUDA call in the process returns an error and only restarting helps. So when you see a strange error, the real cause is usually in some earlier kernel. While debugging, set the environment variable `CUDA_LAUNCH_BLOCKING=1` so every launch runs synchronously and the error is reported on the offending line.

**Check the return value of every CUDA API call.** This is the most important habit in CUDA code, and a detail interviewers notice when reading it.

### compute-sanitizer: out-of-bounds and race checking {#compute-sanitizer越界和竞争检查}

The CUDA Toolkit's `compute-sanitizer` is the GPU's AddressSanitizer:

```bash
compute-sanitizer ./vector_add                      # memcheck by default: out of bounds, misaligned, illegal addresses
compute-sanitizer --tool racecheck ./app            # data races in shared memory
compute-sanitizer --tool initcheck ./app            # reads of uninitialized device memory
compute-sanitizer --tool synccheck ./app            # misuse of the synchronization primitives
```

The program runs tens of times slower but pinpoints the offending thread and source line (compile with `-lineinfo`). **Run a new kernel through memcheck on small data before anything else**; it saves a great deal of debugging.

## Making the numbers trustworthy {#让测出来的数字可信}

You change a kernel and the time drops — how do you know it is really faster? The points below come before any optimization technique.

**State the scope first.** Does "the time for one vector add" include allocating device memory, copying the data, checking the result? Both scopes are useful, but you have to say which one you mean before comparing. The convention throughout this handbook is: **inputs and outputs already live in device memory, the kernel is run repeatedly after a warm-up, and CUDA events measure that window**; allocation, initialisation and `check_close` are outside it.

**There is more than a kernel between two events.** `cudaEventElapsedTime` gives the difference between two timestamps on the GPU's timeline. The GPU executes in submission order: if the queue is empty when the start marker completes and the CPU has not yet submitted the kernel, that idle gap is counted too. So **the shorter the kernel, the larger the share of launch latency inside the measurement**; to see the kernel's own start and end, use [Nsight](../tools/profiling.md). This is also the usual reason "my event timing is larger than what ncu reports".

**Two statistical conventions.** `common.cuh` has two timing helpers, and they do not measure the same thing:

```cuda
float ms = time_ms(run, 20, 3);                     // two events around 20 calls, total / 20: a batch average
auto [min_ms, med_ms] = time_stats(run, 50, 5);     // time each call, keep 50 samples, take the minimum and the median
```

- the **batch average** saves 19 event insertions and synchronisations, but a single total **cannot** be turned back into the distribution of individual calls;
- **per-call samples** show the spread: the **median** is not moved by a few unusually slow samples, and the **minimum** is a best-case reference, not a speed you should claim is reached every time;
- the three numbers (mean, median, minimum) are not interchangeable. Every figure in this handbook comes from `time_ms`, that is, a batch average, which is what makes them comparable with each other.

**Lock the clocks.** A GPU's SM and memory clocks drift with temperature and the power limit, so without locking them the same program can differ by a few percent between two runs:

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS | head -20      # which clocks this card supports
sudo nvidia-smi -lgc 1500,1500                    # lock the SM clock at 1500 MHz
sudo nvidia-smi --lock-memory-clocks=9501,9501    # lock the memory clock too (optional)
# ... run the benchmark ...
sudo nvidia-smi -rgc && sudo nvidia-smi -rmc      # restore afterwards
```

Nsight Compute does something similar for you by default (`--clock-control base` locks the base clock, `--cache-control all` flushes the cache before each replay), at the cost of absolute timings that differ from a normal run.

**Effective bandwidth is not the traffic measured on the memory bus.** Our `gbps(bytes, ms)` divides the **bytes the algorithm requires** (how many arrays were read and written) by the time. With a small amount of data, most of those bytes may hit L2 and never reach the memory bus, so the "effective bandwidth" comes out absurdly high — in the measurement below it reaches 195% of the theoretical peak. Therefore:

- to measure memory bandwidth, make the **working set much larger than L2** (tens of megabytes and up), and report the array size;
- when a number looks suspiciously high, suspect the cache before congratulating yourself; real DRAM traffic is what Nsight's counters such as `dram__bytes` report.

**Do not mix units.** `MiB` is $2^{20}$ bytes and `GB` is $10^9$ bytes. Work out the exact byte count first and then convert to GB/s — dividing MiB by milliseconds directly is off by 4.9%.

**Validate outside the timing window**, and really validate: `check_close` also fails on NaN (that is what the negated comparison is for), otherwise a kernel that produces nothing but NaN can look wonderfully fast.

The points above turn into one benchmark you can run directly. It prints the environment and then does three experiments: three statistical conventions on one kernel, the effective bandwidth of copying arrays of different sizes, and the floor on the time of a kernel that does nothing.

```cuda title="bench_timing.cu"
// bench_timing.cu — the three experiments of this section: the conventions, the working set and the cache, and launch latency on short kernels
// build: nvcc -O3 -arch=sm_75 bench_timing.cu -o bench_timing
// on your own card prefer -arch=native, which saves the JIT on the first run
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
  double peak = 2.0 * mem_khz * 1e3 * bus / 8.0 / 1e9;   // DDR-style memory transfers twice per clock
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

// Experiment 1: one kernel, three statistical conventions, three numbers
static void experiment_scope() {
  const long long n = 4096LL * 4096;                      // 16M elements; the three arrays are 192 MiB together
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

  float batch = time_ms(run, 200, 20);                    // two events around 200 calls, total / 200
  auto [lo, med] = time_stats(run, 200, 20);              // 200 samples, one per call
  double moved = 3.0 * n * sizeof(float);                 // the algorithm's bytes: two arrays read, one written
  std::printf("--- 实验一：同一个向量加法，三种口径 ---\n");
  std::printf("%-18s %10s %12s\n", "统计量", "毫秒", "GB/s");
  std::printf("%-18s %10.6f %12.1f\n", "整批平均", batch, gbps(moved, batch));
  std::printf("%-18s %10.6f %12.1f\n", "逐次中位数", med, gbps(moved, med));
  std::printf("%-18s %10.6f %12.1f\n", "逐次最小值", lo, gbps(moved, lo));

  CUDA_CHECK(cudaMemcpy(hc.data(), c, bytes, cudaMemcpyDeviceToHost));
  std::vector<float> ref(n, 3.f);                         // validation sits outside the timing window
  std::printf("校验：");
  check_close(hc.data(), ref.data(), n);
  std::printf("\n");
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
}

// Experiment 2: copy arrays of different sizes and watch how "effective bandwidth" moves with the working set
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

// Experiment 3: the shorter the kernel, the larger the share of launch latency in the measurement
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

```text title="输出"
GPU: NVIDIA GeForce RTX 5070 Ti (sm_120, 70 SM, L2 48.0 MiB)
SM 时钟 2572 MHz, 显存 14001 MHz x 256 bit -> 理论峰值带宽 896 GB/s
CUDA runtime 13.2, driver 13.4

--- 实验一：同一个向量加法，三种口径 ---
统计量                        毫秒         GB/s
整批平均                 0.264611        760.8
逐次中位数                0.265664        757.8
逐次最小值                0.255104        789.2
校验：PASS  max_abs_err=0.000e+00  mismatches=0/16777216

--- 实验二：工作集大小与有效带宽（复制，一读一写）---
   单数组 MiB       读写 MiB      中位 ms       GB/s      占峰值 校验
         1            2   0.006720      312.1      35% PASS
         4            8   0.009632      870.9      97% PASS
        16           32   0.019232     1744.7     195% PASS
        64          128   0.182816      734.2      82% PASS
       256          512   0.726144      739.3      83% PASS

--- 实验三：一个什么都不做的 kernel ---
整批平均 0.005477 ms，逐次中位数 0.005888 ms，逐次最小值 0.003232 ms
逐次测量的这个下限就是「排队 + 提交」的开销；kernel 自身耗时接近它时，event 的读数基本在量开销
```

<small>Measured on 2026-10-08 on an RTX 5070 Ti (sm_120, 70 SMs, 48 MiB of L2), CUDA runtime 13.2 / driver 13.4, `-arch=native`. The SM clock on the second line is the card's **peak** clock, not the instantaneous one.</small>

The three tables pin down the points above one by one:

- **Experiment 1**: three conventions give three numbers (760.8 / 757.8 / 789.2 GB/s), 1% to 4% apart. The 192 MiB working set is far larger than the 48 MiB of L2, so this really is going to memory, and **85% to 88% of peak** is the normal ceiling for a streaming kernel;
- **Experiment 2**: the effective bandwidth rises and then falls, and the turning point is exactly the size of L2. The 16 MiB array (32 MiB read and written, **less than the 48 MiB of L2**) comes out at 1744.7 GB/s, which is **195%** of the theoretical peak — no card's memory runs at twice its peak, and most of those bytes never left L2. Once the working set reaches 128 MiB and 512 MiB, the numbers fall back to 734 and 739 GB/s, which is what the memory really looks like;
- **The 1 MiB row is only 35%**, and experiment 3 says why: the whole kernel takes 6.7 µs, while a kernel that does **nothing at all** has a per-call floor of 3.2 µs. More than half of that row is measuring "queue plus submit", not bandwidth. That is direct evidence for "the shorter the kernel, the larger the share of launch latency in the measurement".

So one and the same "effective bandwidth" yields 35%, 195% and 83%, all from correct code and correct timing — only the working set changed. **Always report the array size with the number**, or the metric means nothing.

**Write it down or you cannot check it.** Five things worth recording for a benchmark: the machine and the date (GPU model, driver, CUDA version), the full command, the input size and how it was validated, the timing scope and the statistic, and **which statements are observations and which are explanations you have not verified yet**. "Bandwidth is higher for small arrays" is an observation; "because it hits L2" is an explanation to be checked — the next step is to look for evidence in the profiler, not to write it into the conclusion.

## The grid-stride loop {#grid-stride-loop}

The version above needs at least as many threads as elements. Another common shape is the **grid-stride loop**: launch a fixed number of threads and have each loop over several elements with the whole grid's thread count as the stride:

```cuda title="grid_stride.cu"
// grid_stride.cu - SAXPY as a grid-stride loop: y = a * x + y
// build: nvcc -O3 -arch=sm_75 grid_stride.cu -o grid_stride
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

  // the block count follows the GPU's size, not the data's: a few blocks per SM
  const int threads = 256;
  const int blocks = sm_count() * 8;
  saxpy<<<blocks, threads>>>(n, a, d_x, d_y);
  CUDA_CHECK_LAST();
  std::vector<float> h_out(n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  check_close(h_out.data(), ref.data(), n);

  // the timing loop keeps updating y; only the speed matters here
  float ms = time_ms([&] { saxpy<<<blocks, threads>>>(n, a, d_x, d_y); });
  std::printf("saxpy (grid-stride, %d blocks): %.3f ms, %.1f GB/s\n", blocks, ms, gbps(3.0 * bytes, ms));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return 0;
}
```

What a grid-stride loop buys:

- it handles **data of any size**, with no constraint from the grid;
- the block count can follow the GPU's size, so **fewer threads mean less startup and teardown cost**, which is often faster when each element is little work;
- launching the same kernel as `<<<1, 1>>>` makes it serial, which is handy for debugging.

`__restrict__` tells the compiler the pointers do not alias, which allows more aggressive optimization, such as routing read-only data through the read-only cache. Add it whenever you know the pointers are distinct.

## Querying the device {#查询设备信息}

Before writing a kernel, find out what the GPU in front of you has:

```cuda title="device_query.cu"
// device_query.cu - print the GPU's key parameters
// build: nvcc -O3 -arch=sm_75 device_query.cu -o device_query
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
    // DDR-style memory transfers twice per clock
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

`sharedMemPerBlock` is the default ceiling (48 KB) and `sharedMemPerBlockOptin` is the ceiling after an explicit request, which the GEMM chapter will use.

!!! interview "How to explain it"
    This chapter usually shows up as "write a vector addition and explain the details": `__global__` runs on the device and is launched from the host; a launch is asynchronous, so check launch errors with `cudaGetLastError` and wait with a synchronization or an event; guard the bounds when the length is not a multiple of the block size, and mind 64-bit index overflow; a grid-stride loop decouples the kernel from the data size. Time with CUDA events, warm up first, average over several runs; and run a new kernel through `compute-sanitizer`. Volunteering these engineering habits counts for more than merely writing a correct kernel.

## Exercises {#练习}

**1. Two-dimensional matrix addition.** Write a kernel computing `C = A + B` for a `rows × cols` matrix stored row-major. Use a two-dimensional block (`dim3 block(32, 8)`) and a two-dimensional grid. Think about it: should `threadIdx.x` be the row or the column? Why?

??? success "Answer"
    ```cuda title="matrix_add.cu"
    // matrix_add.cu - organizing threads in two dimensions
    // build: nvcc -O3 -arch=sm_75 matrix_add.cu -o matrix_add
    #include "common.cuh"

    __global__ void matrix_add(const float* A, const float* B, float* C, int rows, int cols) {
      int col = blockIdx.x * blockDim.x + threadIdx.x;  // x is the column: neighbouring threads touch neighbouring addresses
      int row = blockIdx.y * blockDim.y + threadIdx.y;
      if (row < rows && col < cols) {
        size_t idx = static_cast<size_t>(row) * cols + col;
        C[idx] = A[idx] + B[idx];
      }
    }

    int main() {
      const int rows = 3000, cols = 5000;  // a size deliberately not divisible
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

    `threadIdx.x` should be the **column**, the innermost, contiguously stored dimension. The 32 threads of a warp have consecutive `threadIdx.x` and touch 32 consecutive elements of one row, so their accesses coalesce. Map x to the row instead and a warp touches 32 different rows, each access crossing a whole row, and bandwidth drops several times over. The next chapter explains why in detail.

**2. Find the bugs.** The program below is correct at `n = 1000` but on some GPUs gives random results or errors at large `n`. Find at least 3 problems.

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

??? success "Answer"
    1. **No bounds check**: `n / 1024` rounds down and there is no `if (i < n)`, so it either misses elements or runs past the end.
    2. **A block size of 2048 is over the 1024 limit**: the kernel never launches at all, but since no error is checked the program "succeeds" silently.
    3. **No CUDA call's return value is checked**: a failing `cudaMalloc` (out of memory, say) goes unnoticed.
    4. **The memory is neither initialized nor freed**: what `cudaMalloc` returns holds undefined contents.
    5. Here `n` is about 750 million, still within `int`; but three times larger and both `int i` and the `n` in `n * sizeof(float)` overflow. Large data should use `size_t` / `int64_t` throughout.

**3. Catch an out-of-bounds access with compute-sanitizer.** Delete the `if (i < n)` from `vector_add`, set `n` to something not divisible by 256, and run it both directly and under `compute-sanitizer`, comparing the output.

??? success "Answer"
    Run directly, the out-of-bounds write very likely lands in the alignment padding of the `cudaMalloc` block and the program often **looks fine**. That is exactly what makes out-of-bounds bugs dangerous. `compute-sanitizer` reports something like `Invalid __global__ write of size 4 bytes` along with the thread and the address, and with `-lineinfo` at compile time it even gives the source line.

## Summary {#小结}

- [x] Check the return value of every CUDA call, and check launch errors with `cudaGetLastError` after a launch.
- [x] A launch is asynchronous; time with CUDA events, warm up first, and average over several runs.
- [x] State the convention (the timing scope and the statistic) before comparing; queue gaps and launch latency also fall between two events, and the shorter the kernel the more they matter; lock the clocks for comparable results.
- [x] Effective bandwidth is 「the algorithm's bytes / time」, and with a small working set it may only be measuring L2; validate outside the timing window.
- [x] Mind the bounds check and 64-bit overflow in index arithmetic.
- [x] Run a new kernel through `compute-sanitizer` first.
- [x] A grid-stride loop decouples the kernel from the data size and is the standard shape in many libraries.
