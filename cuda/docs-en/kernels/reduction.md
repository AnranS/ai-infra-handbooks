# Reduction

<p class="lead">A reduction (summing an array, or taking its maximum) is the classic hand-written CUDA interview question. It looks simple, but getting from the naive version to saturating bandwidth means working through divergence, bank conflicts, synchronization cost, thread utilization and too few outstanding memory requests, which covers nearly everything from the earlier chapters.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is wrong with `if (tid % (2*s) == 0)` in a shared-memory tree reduction?
    2. Why does switching from "interleaved addressing" to "sequential addressing" remove the bank conflicts?
    3. Why is it so much faster to have each thread accumulate several elements serially before the tree reduction?
    4. Why can the last 32 elements be reduced without `__syncthreads()`? What is the modern way?
    5. How are several blocks' partial sums combined? What are the trade-offs?

??? success "Answers (try it yourself first, then expand)"
    1. Only threads whose `tid` is a multiple of `2s` work: half of each warp sits idle while still going along, which is divergence; and the active threads' indices are far apart, which conflicts in shared memory.
    2. Sequential addressing has the first half of the threads add the second half (`sdata[tid] += sdata[tid + s]` with `tid < s`), so the active threads form one contiguous run and a warp is either fully active or fully idle; and neighbouring threads touch neighbouring addresses, with no bank conflict.
    3. In a tree reduction most threads go idle very quickly and every round needs a synchronization. Have each thread accumulate many elements serially in registers first (with vectorized loads) and far fewer threads saturate the bandwidth, with the tree part's cost spread thin.
    4. Those 32 elements are in one warp, whose threads execute together (with `__syncwarp` or volatile), so the whole block need not synchronize. The modern way is `__shfl_down_sync`, exchanging registers directly without shared memory.
    5. Launch a second kernel to reduce the partial sums (deterministic, at the cost of another launch); or have each block atomically add its partial sum to the global result (one launch, but the floating-point result is not reproducible); or have "the last block to finish" do the summing.

## The problem and the performance target {#问题与性能目标}

Given n floats, produce their sum. Each element is read once and added once, so the **arithmetic intensity is minimal and this is purely bandwidth-bound**. The only measure of a reduction kernel is therefore **effective bandwidth**: `n × 4 bytes / time`, aiming close to peak memory bandwidth (85-90% of peak in practice is very good).

The overall shape is two steps: each block computes its own partial sum, then the partial sums are added. The second step moves very little data, so the focus below is on the first.

## Seven versions {#七个版本的演进}

The complete code is at the end of this section; here is what each version changes and why. The first two differ entirely in "which threads work at each step":

![Figure: interleaved against sequential addressing](../assets/figures/reduction-addressing.svg){.aig-svg}

### v0: interleaved addressing, with divergence {#v0交错寻址分支发散}

```cuda
for (unsigned s = 1; s < blockDim.x; s *= 2) {
  if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
  __syncthreads();
}
```

The first round works on even threads, the second on multiples of 4, and so on. More than half of every warp sits idle yet still goes along, which is severe **divergence**; and the modulo itself is slow.

### v1: contiguous threads do the work {#v1连续的线程做事}

```cuda
for (unsigned s = 1; s < blockDim.x; s *= 2) {
  unsigned idx = 2 * s * tid;
  if (idx < blockDim.x) sdata[idx] += sdata[idx + s];
  __syncthreads();
}
```

Now the first few threads by index do the work, so for the first rounds a warp is either entirely working or entirely idle and the divergence is gone. But the access pattern becomes a stride of `2s`: at `s = 16`, thread k touches `sdata[32k]`, all in one bank, a severe **bank conflict**.

### v2: sequential addressing {#v2顺序寻址}

```cuda
for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
  if (tid < s) sdata[tid] += sdata[tid + s];
  __syncthreads();
}
```

Each round adds the second half onto the first. Thread k touches `sdata[k]` and `sdata[k + s]`, so neighbouring threads touch neighbouring addresses: **no bank conflict** and no divergence. This is the standard shape of a shared-memory tree reduction.

### v3: one addition during the load {#v3加载时先做一次加法}

v2 leaves half the threads idle in its very first round. Have each block handle twice the data, with each thread loading two elements and adding them before the tree starts. The block count halves and thread utilization doubles.

### v4: shuffles for the last warp {#v4最后一个-warp-用-shuffle}

Once `s ≤ 32` only one warp is working, yet every round still synchronizes the whole block. Use warp shuffles for the final 64-to-1 reduction and save the last 5 block synchronizations and 5 shared-memory round trips:

```cuda
for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) { ... }
if (tid < 32) {
  float v = sdata[tid] + sdata[tid + 32];
  v = warp_reduce_sum(v);            // 5 rounds of __shfl_down_sync
  if (tid == 0) out[blockIdx.x] = v;
}
```

!!! note "The old `volatile` version"
    The classic material (Mark Harris's 2007 *Optimizing Parallel Reduction in CUDA*) unrolls the last warp with a `volatile` shared-memory pointer and no synchronization, relying on "threads in a warp execute in lockstep". Since Volta's independent thread scheduling that is no longer guaranteed. Use `__shfl_down_sync`, or put `__syncwarp()` between the steps.

### v5: several elements per thread, plus vectorization {#v5每个线程处理多个元素--向量化}

In the versions above each thread handles only one or two elements yet still goes through a whole tree reduction (`log2(256) = 8` synchronizations), so **the synchronization and reduction cost far exceeds the useful additions**; and with only one or two accesses in flight per thread it cannot saturate bandwidth, as the [latency hiding](../basics/execution.md#延迟掩盖) section explained.

v5 does three things:

1. **a grid-stride loop**: launch only "SMs × a few" blocks and have each thread accumulate hundreds or thousands of elements serially in registers, so the reduction's cost becomes negligible;
2. **`float4` vectorized loads**: 16 bytes per memory instruction, four times as much data in flight as the scalar version;
3. **a two-level shuffle reduction**: shuffle within a warp, write each warp's result to shared memory, and have warp 0 shuffle again. Only one `__syncthreads()` is needed.

This version usually reaches above 85% of peak bandwidth and is the standard structure of a production reduction kernel. The "reduce along a row" in softmax and LayerNorm uses the same pattern.

### v6: cub::DeviceReduce {#v6cubdevicereduce}

CUDA ships CUB, whose device-level reduction is heavily optimized and is what to use in production. It is also the baseline for judging your own version:

```cuda
size_t temp_bytes = 0;
cub::DeviceReduce::Sum(nullptr, temp_bytes, d_in, d_out, n);   // the first call only asks how much temporary space is needed
cudaMalloc(&d_temp, temp_bytes);
cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);     // the real run
```

## The complete code {#完整代码}

```cuda title="reduction.cu"
// reduction.cu - seven versions of a reduction, each verified and measured for effective bandwidth
// build: nvcc -O3 -arch=sm_75 reduction.cu -o reduction
#include "common.cuh"
#include <cub/cub.cuh>

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

// v0: interleaved addressing with a modulo test, badly divergent within a warp
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

// v1: contiguous threads work, so no divergence, but the strided access conflicts in banks
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

// v2: sequential addressing, no divergence and no bank conflicts
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

// v3: each block handles 2 * blockDim elements, with one addition during the load
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

// v4: v3 plus warp shuffles for the final 64 -> 1 (needs blockDim >= 64)
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

// v5: a grid-stride loop, float4 loads and a two-level shuffle reduction
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
  for (int i = n4 * 4 + gtid; i < n; i += nthreads) v += in[i];   // the tail of fewer than 4

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
  int elems_per_block;   // 0 means a fixed grid size (the grid-stride loop)
};

int main() {
  const int n = (1 << 26) + 3;   // about 67 million elements, deliberately not a multiple of 4
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
    for (float p : partial) sum += p;   // step two: there are few partial sums, so they are combined on the CPU here
    bool good = std::fabs(sum - ref) <= 1e-5 * ref;
    ok &= good;
    float ms = time_ms([&] { ver.kernel<<<blocks, kThreads>>>(d_in, d_partial, n); });
    std::printf("%-28s %10.3f %10.1f  %s\n", ver.name, ms, gbps(bytes, ms), good ? "PASS" : "FAIL");
  }

  // v6: CUB's device-level reduction
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

Run it and fill in the table below, working out each version's percentage of your GPU's peak bandwidth. The table itself makes a good study record and something to talk through in an interview:

| Version | Main change | Time | Bandwidth | % of peak |
| --- | --- | --- | --- | --- |
| v0 | baseline | | | |
| v1 | divergence removed | | | |
| v2 | bank conflicts removed | | | |
| v3 | one add during the load | | | |
| v4 | warp shuffles | | | |
| v5 | several elements + vectorization | | | |
| v6 | CUB | | | |

## Combining the blocks' partial sums {#合并各-block-的部分和}

| Method | How | Pros | Cons |
| --- | --- | --- | --- |
| two kernels | the first writes the partial sums, the second (one block) reduces them | simple and deterministic | one extra launch |
| atomic add | each block `atomicAdd`s its result when done | one kernel | the floating-point result is not reproducible; the result needs zeroing first |
| the last block sums | each block writes its partial sum, does a `__threadfence()` and atomically increments a counter, and the block with the last ticket does the summing | one kernel and deterministic | somewhat more code |

CUB uses two kernels and guarantees a deterministic result.

## How to explain it {#怎么讲清楚}

When asked to "write a sum kernel", a good rhythm is:

1. write a correct v2-style version (sequential addressing) first, explaining why not interleaved;
2. point out its problems yourself: too little work per thread, too much synchronization;
3. move to the v5 structure of grid-stride plus warp shuffles, explaining what `__shfl_down_sync` and its mask mean;
4. discuss how to combine the blocks' results and floating-point determinism;
5. note that this is a bandwidth-bound kernel measured by effective bandwidth as a fraction of peak, and give numbers you have measured.

!!! interview "How to explain it"
    A reduction is the classic hand-written kernel exercise: it is bandwidth-bound and measured by effective bandwidth as a fraction of peak. The order of optimization: sequential addressing removes divergence and bank conflicts, then each thread accumulates many elements serially in registers (with vectorized loads for more requests in flight, which is the crucial step), then a warp shuffle reduction, and finally the blocks' partial sums are combined with a second kernel or an atomic. Finish by noting that the "warp shuffle, shared memory, shuffle again" two-level structure is the template for every row-wise reduction kernel (softmax, RMSNorm), with CUB as the production baseline.

## Exercises {#练习}

**1. Maximum and its index (argmax).** Modify v5 to output the index of the largest element. Hint: shuffle the value and the index together, and on ties take the smaller index.

??? success "Answer"
    ```cuda title="argmax.cu"
    // argmax.cu - reducing the value and the index together with warp shuffles
    // build: nvcc -O3 -arch=sm_75 argmax.cu -o argmax
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
      h[7'654'321] = 3.f;   // a unique maximum
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

**2. A deterministic single-kernel sum.** Implement the sum in one kernel with the "last block sums" method, requiring bit-identical results across runs.

??? success "Approach"
    1. each block computes its partial sum and writes `partial[blockIdx.x]`;
    2. thread 0 executes `__threadfence()` so the partial sum is visible to other blocks, then `unsigned ticket = atomicAdd(&counter, 1)`;
    3. broadcast "am I the last one" (`ticket == gridDim.x - 1`) to the whole block through shared memory;
    4. the last block reads all the partial sums in a **fixed order**, reduces them, writes the result, and zeroes the counter for next time.

    Because the summing order is fixed (by block index), the result does not depend on the order the blocks finished and is deterministic. CUDA's official `threadFenceReduction` sample does exactly this.

## Summary {#小结}

- [x] A reduction is bandwidth-bound and measured by effective bandwidth as a fraction of peak.
- [x] Sequential addressing removes divergence and bank conflicts; the last warp uses shuffles.
- [x] The crucial optimization is having each thread accumulate many elements in registers, with vectorized loads for more requests in flight.
- [x] The "warp shuffle, shared memory, shuffle again" two-level structure is the template for every row-reduction kernel.
- [x] Use CUB in production, and measure your own version against it.
