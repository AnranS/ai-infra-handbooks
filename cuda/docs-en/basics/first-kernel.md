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
    5. Record CUDA events before and after on the same stream and take the interval with `cudaEventElapsedTime`; warm up first (the first call carries initialization overhead) and take a mean or median over several runs. Never wrap an asynchronous launch in a CPU timer.

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
- [x] Mind the bounds check and 64-bit overflow in index arithmetic.
- [x] Run a new kernel through `compute-sanitizer` first.
- [x] A grid-stride loop decouples the kernel from the data size and is the standard shape in many libraries.
