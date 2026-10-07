# Synchronization, atomics and warp programming

<p class="lead">As soon as threads cooperate, synchronization and atomics come in. This chapter covers the rules for synchronizing inside a block, the correct use and the cost of atomics, and the warp-level primitives: shuffles, votes and cooperative groups. The warp shuffle is the building block of almost every high-performance kernel, from reductions to softmax to attention.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What happens if you put `__syncthreads()` inside `if (threadIdx.x < 16)`?
    2. Ten million threads `atomicAdd` to the same global variable. Why is that slow? How do you improve it?
    3. What does `__shfl_down_sync(0xffffffff, v, 16)` do? What is the first argument?
    4. How do 5 shuffles sum 32 values across a warp? How do you get the sum into all 32 threads?
    5. Can different blocks synchronize with each other?

??? success "Answers (try it yourself first, then expand)"
    1. Only the first 16 threads reach the `__syncthreads()` and the others never do, so the barrier never completes: undefined behaviour (possibly a deadlock, possibly a silent wrong answer). Every thread of a block must execute the barrier.
    2. Every thread contends for one address, so the atomics complete one at a time. Reduce within a warp with shuffles first, then within the block through shared memory, and finally do one global atomic add per block.
    3. Each thread takes `v` from the lane 16 above it (keeping its own value when that is past the end of the warp); the first argument is the mask of participating threads, and `0xffffffff` means all 32.
    4. Add the values at offsets 16, 8, 4, 2 and 1 with `__shfl_down_sync`, and after 5 steps lane 0 holds the sum; to give all 32 lanes the result, use `__shfl_xor_sync` (a butterfly) instead, or broadcast from lane 0 with `__shfl_sync` at the end.
    5. Not by ordinary means inside a kernel: different blocks may not even be running at the same time, so waiting on each other deadlocks. Either split into two kernels (a kernel boundary is a global synchronization), or use a cooperative launch's grid synchronization, or a thread-block cluster (blocks within one cluster can synchronize).

## Synchronizing inside a block: `__syncthreads()` {#block-内同步__syncthreads}

`__syncthreads()` is a **barrier**: execution continues only once every thread of the block has arrived; and writes to shared and global memory before the barrier become visible to every thread of the block after it.

The commonest pattern is "write shared memory, synchronize, read what others wrote":

```cuda
tile[threadIdx.x] = input[i];     // one element written per thread
__syncthreads();                   // make sure every thread has finished writing
float v = tile[blockDim.x - 1 - threadIdx.x];   // read an element another thread wrote
```

**Two iron rules:**

1. **Every thread of the block must reach the same `__syncthreads()`.** Putting it inside a branch only some threads enter is undefined behaviour, usually showing up as a deadlock or a wrong answer:

    ```cuda
    if (threadIdx.x < 16) {
      __syncthreads();   // wrong: the other threads never arrive
    }
    ```

    An early `return` causes the same problem. At the boundary, have the out-of-range threads "do nothing but still take part in the synchronization" rather than returning.

2. **When shared memory is reused in a loop, synchronize after reading too.** Otherwise the next iteration's writes may overwrite data another thread has not finished reading. That is why GEMM's tiling loop has two `__syncthreads()`; see [GEMM](../kernels/gemm.md).

`compute-sanitizer --tool racecheck` and `--tool synccheck` find shared-memory races and synchronization errors.

## Atomics {#原子操作}

When several threads update the same location, an ordinary read-modify-write loses updates. An atomic makes the read-modify-write indivisible:

```cuda
atomicAdd(&counter, 1);            // returns the value before the update
atomicAdd(&sum, x);                // float works too
atomicMax(&m, v);                  // int / unsigned
atomicCAS(&addr, compare, val);    // compare and swap, from which any atomic can be built
atomicExch(&addr, val);
```

Supported types: `int`, `unsigned` and `unsigned long long` support everything; `float` and `double` support `atomicAdd`; `__half2` and `__nv_bfloat162` have `atomicAdd` too. Other combinations (an atomic max on a float, say) have to be built from `atomicCAS`; see this chapter's exercises.

### What atomics cost {#原子操作的性能}

The atomic itself executes in L2 and has decent throughput. What makes it slow is **contention**: when many threads update one address, those operations can only complete one after another. The remedy is **hierarchical aggregation**:

1. reduce within a warp with shuffles, turning 32 values into 1;
2. reduce within the block through shared memory (or shared-memory atomics);
3. do one global atomic per block.

That brings the number of global atomics down to the number of blocks, and the contention essentially disappears. A histogram is the other classic example: keep a private histogram per block in shared memory and merge into the global one at the end; see `histogram.cu` below.

!!! warning "A floating-point atomic add is not deterministic"
    Floating-point addition is not associative, and atomics complete in a different order every time, so summing with `atomicAdd` **can differ in the last few bits between runs**. When training needs bit-for-bit reproducibility, use a deterministic reduction order instead.

## Warp-level primitives {#warp-级原语}

![Figure: warp divergence, with the two paths of one warp running in turn and only some lanes active in each](../assets/figures/warp-divergence.svg){.aig-svg}

Threads of one warp can read each other's registers without going through shared memory, which is a **shuffle**:

```cuda
T __shfl_sync(unsigned mask, T var, int srcLane, int width = 32);      // read var from srcLane
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = 32);   // read from lane - delta
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = 32); // read from lane + delta
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = 32);    // read from lane ^ laneMask
```

The first argument, `mask`, is the **set of threads taking part**, with bit k set meaning lane k participates. Write `0xffffffff` when the whole warp takes part. Every thread in the mask must execute the instruction. A shuffle is faster than exchanging through shared memory and needs no `__syncthreads()`.

### A warp reduction {#warp-归约}

![Figure: a warp reduction, with each __shfl_down_sync step adding in the value offset lanes away](../assets/figures/warp-reduce.svg){.aig-svg}

Five halving steps of `__shfl_down_sync` leave the sum of 32 values in lane 0:

```cuda
__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset /= 2)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;   // only lane 0 ends with the complete sum
}
```

With `__shfl_xor_sync` instead, each step swaps pairs (a butterfly reduction) and **every lane ends with the complete sum**. In softmax and LayerNorm every thread needs the result, so this version is the common one:

```cuda
__device__ __forceinline__ float warp_allreduce_sum(float v) {
  for (int mask = 16; mask > 0; mask /= 2)
    v += __shfl_xor_sync(0xffffffff, v, mask);
  return v;   // every lane ends with the same result
}
```

Replace `+` with `fmaxf` and it finds the maximum. These two functions recur throughout the later chapters.

### Votes and masks {#投票与掩码}

```cuda
unsigned b = __ballot_sync(0xffffffff, pred);   // bit k is lane k's pred
bool any  = __any_sync(0xffffffff, pred);        // any of them true
bool all  = __all_sync(0xffffffff, pred);        // all of them true
int  cnt  = __popc(b);                           // count the set bits
unsigned active = __activemask();                // the threads actually executing right now
__syncwarp();                                    // synchronize within the warp
```

`__ballot_sync` with `__popc` answers "how many threads satisfy the condition, and how many of them are before me" within a warp, which is the basis of stream compaction; see [prefix sum](../kernels/scan.md).

### A complete example: a two-level sum reduction {#一个完整的例子两级归约求和}

```cuda title="block_sum.cu"
// block_sum.cu - a two-level reduction with warp shuffles and shared memory, one global atomic add per block
// build: nvcc -O3 -arch=sm_75 block_sum.cu -o block_sum
#include "common.cuh"

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

// blockDim.x must be a multiple of 32 and at most 1024
__device__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  v = warp_reduce_sum(v);                    // level one: within a warp
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  const int num_warps = blockDim.x / 32;
  v = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
  if (warp == 0) v = warp_reduce_sum(v);     // level two: warp 0 combines the warps' results
  return v;                                  // only thread 0 holds the block's sum
}

__global__ void sum_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) v += x[i];
  v = block_reduce_sum(v);
  if (threadIdx.x == 0) atomicAdd(out, v);   // one atomic per block
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
  bool ok = check_close(&got, &ref_f, 1, 1e-4f, 0.f);   // float accumulation rounds, so allow a relative error of 1e-4

  float ms = time_ms(run);
  std::printf("sum of %d floats: %.3f ms, %.1f GB/s\n", n, ms, gbps(n * sizeof(float), ms));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

Here each thread first accumulates several elements with a grid-stride loop (no synchronization at all in that step) and then reduces within the block. Having each thread accumulate many elements serially first is what lets a reduction kernel saturate bandwidth; the next chapter, [reduction](../kernels/reduction.md), analyses it in detail.

## Histogram: privatizing in shared memory {#直方图共享内存私有化}

```cuda title="histogram.cu"
// histogram.cu - a 256-bin histogram: global atomics against a private histogram in shared memory
// build: nvcc -O3 -arch=sm_75 histogram.cu -o histogram
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
    atomicAdd(&local[data[i]], 1u);           // a shared-memory atomic, contended only within this block
  __syncthreads();
  for (int b = threadIdx.x; b < kBins; b += blockDim.x)
    if (local[b]) atomicAdd(&hist[b], local[b]);   // at most one global atomic per bin per block
}

int main() {
  const int n = 1 << 26;
  std::vector<uint8_t> h(n);
  std::mt19937 gen(3);
  std::normal_distribution<float> dist(128.f, 20.f);   // concentrated in the middle bins, where contention is worse
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

An integer histogram is exact, so the comparison is for exact equality. Further optimizations: read 4 or 16 bytes per thread at a time; and when the data is highly concentrated, keep several copies of the histogram within one block to spread the contention.

## Cooperative Groups {#cooperative-groups}

Cooperative Groups is CUDA's more structured API for thread groups, turning "a group of threads" into an object and making the intent of the code clearer:

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void k(const float* x, float* out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  float v = x[block.group_index().x * block.size() + block.thread_rank()];
  float s = cg::reduce(warp, v, cg::plus<float>());   // a warp reduction, with every thread getting the result
  if (warp.thread_rank() == 0) atomicAdd(out, s);
  block.sync();                                         // equivalent to __syncthreads()
}
```

It also supports smaller tiles (`tiled_partition<16>` and so on, good for one warp handling several rows of a small matrix), grouping by condition (`coalesced_threads()`), and **synchronizing the whole grid**: `cg::this_grid().sync()`. Grid synchronization requires launching with `cudaLaunchCooperativeKernel` and every block being resident at once, or it deadlocks.

**Blocks should not wait on each other** in principle (apart from the controlled grid synchronization above). When global synchronization is needed, the usual answer is to split the computation into two kernels. Hopper's **thread-block clusters** offer a middle ground: blocks within one cluster can synchronize and read each other's shared memory; see [Hopper](../advanced/async-hopper.md).

!!! interview "Answering in an interview"
    The usual questions on synchronization and warp programming: `__syncthreads()` must be executed by every thread of the block, and putting it in a branch deadlocks or corrupts; atomics are slow because of contention, so aggregate within the warp and the block and then do a few global atomics, and a floating-point atomic add is not reproducible; a shuffle exchanges registers directly, and five `__shfl_xor_sync` butterfly steps give all 32 lanes the sum. Blocks should not wait on each other, so global synchronization means splitting into two kernels, or a cooperative launch or a thread-block cluster.

## Exercises {#练习}

**1. A floating-point atomic maximum.** CUDA has no float `atomicMax`. Implement `atomicMaxFloat(float* addr, float val)` with `atomicCAS`.

??? success "Answer"
    ```cuda title="atomic_max_float.cu"
    // atomic_max_float.cu - a floating-point atomic maximum built from atomicCAS
    // build: nvcc -O3 -arch=sm_75 atomic_max_float.cu -o atomic_max_float
    #include "common.cuh"

    __device__ float atomicMaxFloat(float* addr, float val) {
      int* p = reinterpret_cast<int*>(addr);
      int old = *p;
      while (__int_as_float(old) < val) {
        int assumed = old;
        old = atomicCAS(p, assumed, __float_as_int(val));   // returns assumed on success
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

    The loop means: as long as the current value is below `val`, try to replace it; if another thread changed it between the read and the swap, the new value `atomicCAS` returns goes into the next round. In practice, find the maximum within the warp and the block first and do one atomic at the end.

    Another common trick uses the IEEE bit pattern: a non-negative float keeps its order when reinterpreted as an int, so non-negative values can go straight through `atomicMax((int*)addr, __float_as_int(val))`; negatives need `atomicMin` on the unsigned representation instead.

**2. A prefix sum within a warp.** Write a function with `__shfl_up_sync` that returns each lane's inclusive prefix sum (lane k gets the sum of lanes 0 through k).

??? success "Answer"
    ```cuda
    __device__ __forceinline__ int warp_inclusive_scan(int v) {
      const int lane = threadIdx.x % 32;
      for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, v, offset);
        if (lane >= offset) v += n;   // the first offset lanes have nobody above them
      }
      return v;
    }
    ```

    Five steps, each doubling the range already accumulated (the Hillis-Steele algorithm). The full block-level and device-level scans are in [prefix sum](../kernels/scan.md).

## Summary {#小结}

- [x] `__syncthreads()` must be executed by every thread of the block, and reusing shared memory in a loop needs a synchronization after the read too.
- [x] Atomics are slow because of contention; aggregate within the warp and the block and do only a few global atomics.
- [x] A floating-point atomic add is not reproducible.
- [x] A shuffle exchanges registers directly within a warp; the `__shfl_xor_sync` butterfly gives every lane the result.
- [x] Blocks should not wait on each other; global synchronization means splitting kernels, or a cooperative launch or a thread-block cluster.
