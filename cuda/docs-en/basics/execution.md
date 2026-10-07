# The execution model and performance fundamentals

<p class="lead">Knowing what the hardware looks like, the next question is how it runs your code: how warps are scheduled, why a branch can be slow, what occupancy really means, and why "more threads" is not always faster. The roofline model ties it all together into a way of judging a kernel's performance ceiling.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which of `if (threadIdx.x % 2 == 0)` and `if ((threadIdx.x / 32) % 2 == 0)` causes divergence?
    2. Is higher occupancy always better?
    3. Why is it hard to saturate memory bandwidth when each thread has only one outstanding request?
    4. What is arithmetic intensity? How does it tell you whether a kernel is memory-bound or compute-bound?
    5. What happens when you launch 109 blocks on a GPU with 108 SMs (each holding only one)?

??? success "Answers (try it yourself first, then expand)"
    1. The first does: odd and even threads of one warp take different paths and the two run one after the other. The second does not: `threadIdx.x / 32` is the same across a warp, so the whole warp takes one path.
    2. No. Occupancy is a means of hiding latency: enough is enough, and more is not always faster. Sometimes spending more registers (and lowering occupancy) buys instruction-level parallelism and data reuse that make it faster.
    3. Bandwidth = bytes in flight ÷ latency (Little's law). Device-memory latency is hundreds of cycles, and with one outstanding request per thread the whole GPU cannot keep enough bytes in flight to fill the bandwidth; each thread has to issue several independent accesses (unrolling, vectorized loads).
    4. Arithmetic intensity = floating-point operations ÷ bytes moved. Compare it with the hardware's ridge point (peak compute ÷ bandwidth): below it the kernel is memory-bound, above it compute-bound.
    5. 108 blocks run in the first wave and the remaining 1 takes a wave of its own, during which the GPU is almost empty, so the total is close to two waves. That is the tail-wave effect: keep the block count a multiple of the SM count, or use smaller blocks and more waves to dilute it.

## Branch divergence {#分支发散}

A warp's 32 threads share one instruction stream. At a conditional branch, if threads **within one warp** go different ways, the hardware runs each path in turn with the other threads masked off, then reconverges:

```cuda
if (threadIdx.x % 2 == 0) {
  a();          // the even threads run while the odd ones wait
} else {
  b();          // the odd threads run while the even ones wait
}
```

This warp takes the time of `a()` plus `b()`, not the larger of the two. The key is that **divergence only happens inside a warp**. If the condition is the same for every thread of a warp, there is no cost:

```cuda
if ((threadIdx.x / 32) % 2 == 0) { a(); } else { b(); }   // grouped by warp, so no divergence
```

A practical way to tell:

- the condition depends only on `blockIdx`, a kernel argument or something warp-uniform like `threadIdx.x / 32`: no divergence;
- a bounds check `if (i < n)`: only the last warp can diverge, which is negligible;
- a data-dependent branch (`if (x[i] > 0)`): possibly severe divergence, so consider a branch-free form such as `y = max(x, 0.f)`, or `fmaxf` and conditional assignment so the compiler emits a select.

The program below compares the two kinds of branch:

```cuda title="divergence.cu"
// divergence.cu - what divergence inside a warp costs
// build: nvcc -O3 -arch=sm_75 divergence.cu -o divergence
#include "common.cuh"

__device__ __forceinline__ float heavy(float x, int iters, float c) {
  for (int k = 0; k < iters; ++k) x = x * c + 0.5f;
  return x;
}

// odd and even threads of one warp take different branches
__global__ void divergent(float* out, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = static_cast<float>(threadIdx.x);
  if (threadIdx.x % 2 == 0) x = heavy(x, iters, 0.999f);
  else                      x = heavy(x, iters, 0.998f);
  out[i] = x;
}

// the condition is warp-uniform, so a warp takes one path
__global__ void uniform(float* out, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = static_cast<float>(threadIdx.x);
  if ((threadIdx.x / 32) % 2 == 0) x = heavy(x, iters, 0.999f);
  else                             x = heavy(x, iters, 0.998f);
  out[i] = x;
}

int main() {
  const int threads = 256, blocks = sm_count() * 8, iters = 2000;
  const int n = threads * blocks;
  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

  // correctness: both kernels do exactly the same work per thread, only grouped differently
  std::vector<float> h(n), ref(n);
  uniform<<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int t = 0; t < n; ++t) {
    int tid = t % threads;
    float x = static_cast<float>(tid), c = ((tid / 32) % 2 == 0) ? 0.999f : 0.998f;
    for (int k = 0; k < iters; ++k) x = x * c + 0.5f;
    ref[t] = x;
  }
  bool ok = check_close(h.data(), ref.data(), n, 1e-3f, 1e-3f);

  float t_div = time_ms([&] { divergent<<<blocks, threads>>>(d_out, iters); });
  float t_uni = time_ms([&] { uniform<<<blocks, threads>>>(d_out, iters); });
  std::printf("divergent: %.3f ms\nuniform  : %.3f ms\nratio    : %.2fx\n", t_div, t_uni, t_div / t_uni);
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

Expect `divergent` to take about twice as long as `uniform`.

## Hiding latency {#延迟掩盖}

A result is not available immediately after an instruction issues: about 4-6 cycles for arithmetic, tens for shared memory, hundreds for device memory. A GPU does not reorder a thread's later instructions the way a CPU does; it **switches to another warp** instead. Keeping the compute units busy needs enough "ready" work, which comes from two places:

- **Thread-level parallelism (TLP)**: more warps resident on the SM;
- **Instruction-level parallelism (ILP)**: several independent instructions within one thread that can issue back to back without waiting for the previous result.

A useful estimate is **Little's law**: to sustain bandwidth B, the data in flight (issued but not returned) has to reach `B × latency`. On an A100 that is about 2 TB/s × about 600 ns ≈ 1.2 MB, which spread over 108 SMs is about 11 KB in flight per SM. With one 4-byte read in flight per thread, an SM would need nearly 3000 threads, past the 2048 ceiling.

So **occupancy alone often cannot saturate bandwidth**. The fix is to have each thread issue more accesses at once: read 16 bytes with a `float4`, or have each thread handle several elements and issue the loads together up front. That is why nearly every high-performance kernel later in the book has each thread handle several elements.

## Occupancy {#占用率occupancy}

**Occupancy = warps resident on an SM / the hardware ceiling** (64 warps per SM on an A100 or H100). Whether a block can be resident depends on four resources:

| Resource | A100 ceiling per SM | How it limits |
| --- | --- | --- |
| threads | 2048 | the larger the block, the fewer fit |
| blocks | 32 | a very small block (32 threads) hits this one first |
| registers | 65536 | registers per thread × threads |
| shared memory | 164 KB | shared memory per block |

The strictest of the four wins. Move these four numbers around and see which runs out first:

<div class="aig-widget" data-widget="occupancy"></div>

CUDA also computes it for you:

```cuda title="occupancy.cu"
// occupancy.cu - computing occupancy for different configurations with the occupancy API
// build: nvcc -O3 -arch=sm_75 occupancy.cu -o occupancy
#include "common.cuh"

__global__ void light_kernel(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * 2.f + 1.f;
}

// each block uses dynamic shared memory, to imitate a shared-memory limit
__global__ void smem_kernel(float* x, int n) {
  extern __shared__ float buf[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  buf[threadIdx.x] = (i < n) ? x[i] : 0.f;
  __syncthreads();
  if (i < n) x[i] = buf[blockDim.x - 1 - threadIdx.x];
}

template <typename K>
void report(const char* name, K kernel, int block, size_t smem) {
  int dev = 0, blocks_per_sm = 0;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block, smem));
  float occ = static_cast<float>(blocks_per_sm * block) / p.maxThreadsPerMultiProcessor;
  std::printf("%-12s block=%4d smem=%6zu B -> %2d blocks/SM, occupancy %5.1f%%\n",
              name, block, smem, blocks_per_sm, occ * 100);
}

int main() {
  for (int block : {32, 64, 128, 256, 512, 1024}) report("light", light_kernel, block, 0);
  for (size_t smem : {0, 16 * 1024, 32 * 1024, 48 * 1024}) report("smem", smem_kernel, 256, smem);

  // let the runtime suggest a block size (maximizing occupancy)
  int min_grid = 0, best_block = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block, light_kernel, 0, 0));
  std::printf("suggested block size for light_kernel: %d (min grid %d)\n", best_block, min_grid);
  return 0;
}
```

!!! warning "Higher occupancy is not automatically better"
    Occupancy is only one way to hide latency. Many high-performance kernels (GEMM, FlashAttention) deliberately use a lot of registers and shared memory and run at 10-25% occupancy, yet are faster thanks to abundant ILP and data reuse. The rule of thumb: **memory-bound kernels with little work per thread want high occupancy; compute-intensive kernels with plenty of ILP can live with low occupancy**. The criterion is always measured performance, never occupancy itself.

### `__launch_bounds__` {#\_\_launch\_bounds\_\_}

The compiler does not know what block size you will launch with and may allocate so many registers that the configuration you want cannot be resident. Tell it:

```cuda
__global__ void __launch_bounds__(256, 2) my_kernel(...) { ... }
// launched with at most 256 threads per block, aiming for at least 2 resident blocks per SM
```

It then caps the registers per thread, spilling if it must.

## Choosing the launch configuration {#启动配置怎么选}

There is no universal answer, but there are reliable starting points:

- **Block size**: a multiple of 32, starting from 128 or 256. Too small (32, 64) and you hit the per-SM block ceiling first; too large (1024) and scheduling is coarse while `__syncthreads()` makes more threads wait.
- **Grid size**: at least a few blocks per SM. When the data is small and there are fewer blocks than SMs, most of the GPU is idle, and each block should handle less data or parallelism should come from something like split-K (which is exactly the situation for many kernels in an LLM's decode phase).
- **Wave quantization**: one batch of blocks occupying every SM at once is a "wave". Suppose each SM holds 1 block and you launch 109 blocks on 108 SMs: that takes two waves, the second with only 1 block running, which nearly doubles the time. Be careful when the block count is just over a multiple of "SMs × blocks per SM". GEMM libraries adjust their tile sizes for this, or use a load-balancing scheme such as stream-K.

## The roofline model {#roofline-模型}

The roofline model bounds a kernel's performance from two hardware numbers:

- peak compute **P** (FLOP/s)
- peak bandwidth **B** (byte/s)

plus one property of the kernel:

- **arithmetic intensity** I = total compute (FLOP) / total memory traffic (bytes moved to and from device memory)

The ceiling on achievable performance is:

$$
\text{Performance} \le \min(P,\ I \times B)
$$

The two lines meet at the **ridge point** $I^* = P / B$:

- $I < I^*$: **memory-bound**. Optimize by moving fewer bytes (fusion, reuse, quantization) or using the bandwidth better (coalescing, vectorization, more requests in flight).
- $I > I^*$: **compute-bound**. Optimize with faster units (Tensor Cores), less redundant computation, better instruction efficiency.

A few typical kernels (FP32, with the A100's ridge at about 9.6 FLOP/byte; with BF16 Tensor Cores the ridge is about 153):

| Kernel | Arithmetic intensity | Conclusion |
| --- | --- | --- |
| vector add `c = a + b` | 1 FLOP / 12 B ≈ 0.08 | severely memory-bound |
| softmax, LayerNorm | a few to a dozen operations per element / about 8 B | memory-bound |
| matrix-vector GEMV (an M×K matrix) | 2MK / 4MK ≈ 0.5 | memory-bound (the main cost of LLM decode) |
| square GEMM (N×N) | 2N³ / 12N² = N/6 | compute-bound at large N |

This table explains a great deal about LLM inference: **prefill is mostly GEMM and compute-bound, while decode generates one token at a time, which degrades the linear layers into GEMV and makes it memory-bound**. So decode optimization is about reading fewer bytes of weights and KV cache: batching (merging several requests' GEMVs into a GEMM), quantization (weights from 16 bits down to 8 or 4), KV cache compression and so on.

**How to evaluate a kernel with roofline**: compute (or measure with Nsight Compute) its arithmetic intensity, decide whether it is memory-bound or compute-bound, and compare measured performance against the matching ceiling (`I × B` or `P`). Reaching 70-80% of that ceiling is already a very good implementation. Nsight Compute draws the roofline directly; see [profiling](../tools/profiling.md).

## A few instruction-level points {#指令层面的几个要点}

- **FMA**: `a * b + c` compiles to one fused multiply-add (FFMA) and counts as 2 floating-point operations. Peak compute figures are counted in FMAs.
- **Fast math functions**: `__expf`, `__logf`, `__sinf` and `__fdividef` run on the special function units (SFU), much faster than the standard `expf` and friends at slightly lower precision. The `--use_fast_math` option substitutes them globally and also turns on denormal flushing and the like, which is usually fine in training and inference kernels but should be checked for accuracy.
- **Integer division and modulo** are slow; use shifts and masks when the divisor is a power of two, or let the compiler see a compile-time constant.
- **Double precision**: consumer GPUs do FP64 at 1/64 of FP32, and data-centre cards (A100, H100) at 1/2. AI work essentially never uses FP64.

!!! interview "Answering in an interview"
    The usual follow-ups on the execution model: divergence happens only within a warp, so `threadIdx.x % 2` diverges while `(threadIdx.x / 32) % 2` does not; latency is hidden by TLP (more warps) and ILP (more independent operations per thread), and a bandwidth-bound kernel often needs several accesses in flight per thread; occupancy is limited jointly by threads, registers and shared memory, and it is a means rather than a goal. Then mention wave quantization: launch 109 blocks on 108 SMs and the last wave runs one block, nearly doubling the time. Finish with roofline to show "decide the kind of bottleneck first, then pick the optimization".

## Exercises {#练习}

**1. Computing occupancy.** On an H100 (2048 threads, 32 blocks, 65536 registers and 228 KB of shared memory per SM), a kernel has 128 threads per block, 168 registers per thread and 64 KB of shared memory per block. How many blocks fit on an SM? Which resource is the bottleneck?

??? success "Answer"
    - threads: 2048 / 128 = 16 blocks
    - blocks: 32
    - registers: 128 × 168 = 21504 per block, and 65536 / 21504 ≈ 3.05, so 3 blocks
    - shared memory: 228 / 64 ≈ 3.56, so 3 blocks (in practice a small reservation per block comes off too)

    At most 3 blocks, 12 warps, an occupancy of 18.75%. Both registers and shared memory are the bottleneck. This is a common configuration for a high-performance GEMM or attention kernel, which wins through ILP and data reuse.

**2. Finding the bottleneck.** RMSNorm takes a BF16 tensor of `[tokens, hidden]`, reading and writing each element once plus a weight vector of length hidden (negligible), with about 4 floating-point operations per element. On an H100 (3.35 TB/s, 67 TFLOPS of non-Tensor-Core FP32), what bounds it? What is the theoretical minimum time at `tokens = 4096, hidden = 8192`?

??? success "Answer"
    Each element moves 4 bytes in total for about 4 operations, an arithmetic intensity of about 1 FLOP/byte, far below the ridge of 67 / 3.35 = 20, so it is **memory-bound**.

    The data is 4096 × 8192 × 4 B ≈ 134 MB, so the theoretical minimum is 134 MB / 3.35 TB/s ≈ 40 µs. A well-written kernel reaches 80-90% of peak bandwidth, about 45-50 µs. Fusing it with the preceding op (a residual add, say) saves a whole read and write, which is exactly what fusion is worth.

**3. Wave quantization.** A kernel holds only 2 blocks per SM on an A100 and you launch 220 blocks, each taking the same time. How does that compare with launching 216? How would you improve it?

??? success "Answer"
    One wave holds 108 × 2 = 216 blocks. 216 is exactly one wave; 220 needs two, the second with only 4 blocks, so about **twice** the time.

    To improve it: adjust how much data each block handles so the block count is an exact multiple of 216, or far larger than it (the more waves, the smaller the share of the last partial one); or use a persistent kernel that launches exactly 216 blocks, each looping to claim work.

## Summary {#小结}

- [x] Divergence happens only within a warp; keep branch conditions warp-uniform.
- [x] Latency is hidden by TLP (more warps) and ILP (more independent operations per thread); bandwidth-bound kernels often need several accesses in flight per thread.
- [x] Occupancy is limited jointly by threads, blocks, registers and shared memory; it is a means, not a goal.
- [x] Watch for wave quantization and for "too few blocks to fill the GPU".
- [x] Roofline: arithmetic intensity sets the ceiling, so decide the kind of bottleneck first and then pick the optimization.
