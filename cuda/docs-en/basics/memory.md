# The memory hierarchy and access optimization

<p class="lead">For the great majority of kernels the bottleneck is not compute but memory. This chapter covers coalesced global-memory access, shared-memory bank conflicts, vectorized loads, constant memory and register spills. They are the groundwork for every kernel optimization later in the book, and the most frequently tested topic in interviews.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. A warp reads 32 consecutive floats. How few 32-byte sectors does that take? And if each thread reads every other element?
    2. How many banks does shared memory have? When is there a bank conflict? When is there none even though the address is the same?
    3. Why is a two-dimensional shared-memory array so often declared `[32][33]` rather than `[32][32]`?
    4. What does a `float4` vectorized load buy? What does it require?
    5. What is a register spill? How do you spot one?

??? success "Answers (try it yourself first, then expand)"
    1. 32 consecutive floats are 128 bytes, so at least 4 sectors; reading every other element spans 256 bytes, which takes 8 sectors with only half the data useful.
    2. 32 banks, each 4 bytes wide, with addresses falling into banks every 4 bytes. Several threads of a warp touching **different** addresses in the same bank conflict and serialize; touching **the same** address broadcasts and does not conflict.
    3. Reading `[32][32]` by column puts a column's 32 elements 32 floats apart, all in the same bank, a 32-way conflict; one extra element per row (`[32][33]`) staggers a column across 32 different banks.
    4. One instruction reads 16 bytes, cutting the number of memory instructions to a quarter and keeping more data in flight per thread. It requires 16-byte alignment (both the array's start and the index) and either an element count divisible by 4 or a separate tail.
    5. When registers run out, the compiler puts some variables in local memory (really in device memory, through L1), which is slower. Look for spill stores / loads in the `-Xptxas -v` output, or at local-memory traffic in Nsight Compute.

A six-panel strip before the text:

<!-- comic ../assets/comics/cuda-memory.webp is in Chinese; put it back once the English version exists -->

## Global memory: coalescing {#全局内存合并访问}

Global memory is accessed in units of **32-byte sectors**. When a warp executes a memory instruction, the hardware counts how many distinct sectors its 32 addresses fall into and issues that many sector transfers. So:

| Access pattern (one float per thread) | Sectors touched | Useful fraction |
| --- | --- | --- |
| contiguous and aligned: thread k reads `a[base + k]` | 4 (128 bytes) | 100% |
| contiguous but misaligned: the start is offset by one float | 5 | 80% |
| stride 2: thread k reads `a[2k]` | 8 | 50% |
| stride ≥ 8: every thread in its own sector | 32 | 12.5% |
| every thread reads the same address | 1 | broadcast, one transfer |

That is **coalescing**: have a warp's threads touch **contiguous addresses**. The rule is simple, but it decides how much bandwidth you get. The most common way to break it in practice is letting `threadIdx.x` run down the "row" dimension of two-dimensional data, so neighbouring threads are a whole row apart.

![Figure: coalesced versus strided access](../assets/figures/coalescing.svg){.aig-svg}

Change the pattern and see how many bytes these 32 threads make the hardware move:

<div class="aig-widget" data-widget="coalesce"></div>

The program below measures real bandwidth at different strides and offsets. Run it on your own GPU and see the gap for yourself:

```cuda title="access_pattern.cu"
// access_pattern.cu - measuring what stride and misalignment do to bandwidth
// build: nvcc -O3 -arch=sm_75 access_pattern.cu -o access_pattern
#include "common.cuh"

__global__ void strided_copy(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[static_cast<size_t>(i) * stride];
}

__global__ void offset_copy(const float* __restrict__ in, float* __restrict__ out, int n, int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i + offset];
}

int main() {
  const int n = 1 << 22;             // 4M useful elements copied each time
  const int max_stride = 32;
  const size_t in_elems = static_cast<size_t>(n) * max_stride + 64;
  std::vector<float> h_in(in_elems);
  for (size_t i = 0; i < in_elems; ++i) h_in[i] = static_cast<float>(i % 1000);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, in_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256, blocks = (n + threads - 1) / threads;
  std::vector<float> h_out(n), ref(n);
  bool ok = true;

  std::printf("stride  time(ms)  useful GB/s\n");
  for (int stride : {1, 2, 4, 8, 16, 32}) {
    strided_copy<<<blocks, threads>>>(d_in, d_out, n, stride);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) ref[i] = h_in[static_cast<size_t>(i) * stride];
    ok &= check_close(h_out.data(), ref.data(), n, 0.f, 0.f);
    float ms = time_ms([&] { strided_copy<<<blocks, threads>>>(d_in, d_out, n, stride); });
    // count only the useful bytes: n floats read, n written
    std::printf("%6d  %8.3f  %10.1f\n", stride, ms, gbps(2.0 * n * sizeof(float), ms));
  }

  std::printf("\noffset  time(ms)  GB/s\n");
  for (int offset : {0, 1, 2, 4, 8, 16, 32}) {
    float ms = time_ms([&] { offset_copy<<<blocks, threads>>>(d_in, d_out, n, offset); });
    std::printf("%6d  %8.3f  %10.1f\n", offset, ms, gbps(2.0 * n * sizeof(float), ms));
  }
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

You will see effective bandwidth roughly halve as the stride grows from 1 to about 8 and then level off (each thread already has a sector to itself); a misaligned access costs far less, since it only adds a sector or two and L2 makes up part of it.

!!! tip "SoA against AoS"
    An array of structures (AoS, `struct Particle { float x, y, z, m; } p[N]`) puts 16 bytes between neighbouring threads' fields, so reading `p[i].x` wastes 75% of each transfer. A structure of arrays (SoA, `float x[N], y[N], z[N], m[N]`) makes every field's access contiguous. Prefer SoA on a GPU.

## Vectorized loads {#向量化访存}

Reading one `float` per thread needs one 32-bit memory instruction; a `float4` reads 16 bytes at a time and cuts the instruction count to a quarter:

```cuda
__global__ void copy_vec4(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) out[i] = in[i];   // compiles to the 128-bit LDG.E.128 / STG.E.128 instructions
}
```

The gain is fewer issued instructions and less address arithmetic, which usually buys a few percent more bandwidth in a memory-bound kernel and considerably more in kernels where each thread handles several elements (GEMM, normalization). It requires:

- **16-byte aligned addresses**. What `cudaMalloc` returns is aligned to at least 256 bytes, but reading from some offset inside it (a tensor slice, say) is not necessarily aligned;
- a separate tail when the element count is not a multiple of 4.

For `half` the counterparts are `half2` (4 bytes) and `uint4`/`float4` (16 bytes, 8 halves at a time), which FP16/BF16 kernels use constantly.

## Read-only data and the L1 cache {#只读数据与-l1-缓存}

With a `const __restrict__` pointer the compiler knows the data is read-only for the kernel's lifetime and can use the read-only path (`LDG` with a non-coherent cache hint). You can also read explicitly with `__ldg(ptr)`. On modern GPUs L1 and the texture cache are unified and the difference is smaller than it used to be, but **writing `const __restrict__` is a good habit** and helps the compiler with other optimizations too.

## Shared memory {#共享内存}

Shared memory is a block of fast SRAM on each SM, shared by **every thread of one block** and read and written explicitly by you. It has two uses:

1. **Data reuse**: bring in data from global memory that will be used several times and read it from shared memory afterwards, as GEMM tiling does;
2. **Communication between threads**: threads in a block exchange data through it, as in a reduction or a transpose.

Two ways to declare it:

```cuda
// a static size
__global__ void k1() {
  __shared__ float tile[32][33];
}

// a dynamic size: the byte count is the third execution-configuration argument at launch
__global__ void k2() {
  extern __shared__ float buf[];
}
k2<<<grid, block, smem_bytes>>>();
```

A block gets at most 48 KB of shared memory by default. To use more (up to 163 KB on an A100, 227 KB on an H100) you must use dynamic shared memory and request it explicitly before the launch:

```cuda
cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
k2<<<grid, block, 100 * 1024>>>();
```

Shared memory and the L1 cache are the same physical storage. More shared memory means less L1, and the split can be hinted with `cudaFuncAttributePreferredSharedMemoryCarveout`, though leaving it to the driver is usually fine.

### Bank conflicts {#bank-冲突}

Shared memory is split into **32 banks**, each 4 bytes wide. Addresses are handed to banks one 4-byte word at a time: word k belongs to bank `k % 32`. Each bank serves one access per cycle, so:

- a warp's 32 threads touching **32 different banks**: done in one go, the ideal;
- several threads touching **different addresses in the same bank**: those accesses serialize, which is a **bank conflict**. n threads on one bank is an n-way conflict and takes n times as long;
- several threads touching **the same address**: a broadcast, not a conflict.

The classic conflict is reading a two-dimensional array by column:

```cuda
__shared__ float tile[32][32];
float v = tile[threadIdx.x][0];   // thread k reads row k, column 0
```

Row k, column 0 is word `32k`, so every thread lands in bank 0, a 32-way conflict. The fix is **padding by one column**:

```cuda
__shared__ float tile[32][33];    // 33 elements per row
float v = tile[threadIdx.x][0];   // word 33k, which modulo 32 is k, so 32 threads land in 32 different banks
```

Switch between a few patterns and see which banks these 32 threads land on:

<div class="aig-widget" data-widget="bankconf"></div>

The program below measures shared-memory access speed at different strides:

```cuda title="bank_conflict.cu"
// bank_conflict.cu - what a shared-memory bank conflict costs
// build: nvcc -O3 -arch=sm_75 bank_conflict.cu -o bank_conflict
#include "common.cuh"

constexpr int kWords = 32 * 33;

// each thread reads s[lane * STRIDE] over and over
template <int STRIDE>
__global__ void smem_stride(float* out, int iters) {
  __shared__ float s[kWords];
  for (int i = threadIdx.x; i < kWords; i += blockDim.x) s[i] = static_cast<float>(i);
  __syncthreads();

  volatile float* vs = s;  // volatile: forces every iteration to really read shared memory
  const int idx = (threadIdx.x % 32) * STRIDE;
  float acc = 0.f;
  for (int k = 0; k < iters; ++k) acc += vs[idx];
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <int STRIDE>
bool run(float* d_out, int blocks, int threads, int iters) {
  smem_stride<STRIDE><<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK_LAST();
  const int n = blocks * threads;
  std::vector<float> h(n), ref(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int t = 0; t < n; ++t) ref[t] = static_cast<float>(iters) * ((t % threads) % 32 * STRIDE);
  std::printf("stride %2d: ", STRIDE);
  bool ok = check_close(h.data(), ref.data(), n, 1e-6f, 0.f);
  float ms = time_ms([&] { smem_stride<STRIDE><<<blocks, threads>>>(d_out, iters); });
  std::printf("           time %.3f ms\n", ms);
  return ok;
}

int main() {
  const int blocks = sm_count() * 4, threads = 256, iters = 4096;
  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
  bool ok = run<1>(d_out, blocks, threads, iters);   // no conflict
  ok &= run<2>(d_out, blocks, threads, iters);       // 2-way conflict
  ok &= run<4>(d_out, blocks, threads, iters);       // 4-way conflict
  ok &= run<32>(d_out, blocks, threads, iters);      // 32-way conflict
  ok &= run<33>(d_out, blocks, threads, iters);      // padded, so no conflict
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

For 32-bit accesses, a stride of s gives `gcd(s, 32)`-way conflicts: stride 2 is 2-way, stride 32 is 32-way, and stride 33 is coprime with 32 and conflict-free. The Nsight Compute metric `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` counts the conflicts directly; see [profiling](../tools/profiling.md).

!!! tip "Beyond padding: swizzling"
    Padding wastes a little shared memory and breaks 16-byte alignment, which collides with vectorized accesses, `ldmatrix` and TMA, all of which want alignment. High-performance libraries (CUTLASS, FlashAttention) more often use a **swizzle**: xor the column index with the row index, say `col ^ (row % 8)`, to scramble the distribution across banks without wasting space or alignment. The [Tensor Core](../advanced/tensor-core.md) chapter uses it.

## Constant memory {#常量内存}

A `__constant__` variable lives in device memory but has a dedicated constant cache. Its characteristic: **when all threads of a warp read the same address it broadcasts and completes in one go; reading different addresses serializes**. So it suits small parameters every thread reads, such as convolution weights or polynomial coefficients:

```cuda
__constant__ float c_weights[64];
cudaMemcpyToSymbol(c_weights, h_weights, 64 * sizeof(float));   // copied from the host
```

Its capacity is only 64 KB. Kernel arguments themselves are passed through constant memory, which is why passing a small struct by value to a kernel is also efficient.

## Registers and local memory {#寄存器与本地内存}

A thread's local variables go into registers first. They end up in **local memory** (physically in device memory, merely private per thread and cached through L1/L2) when:

- registers run out and the compiler **spills** some variables;
- a local array is indexed with a **value known only at run time** (registers cannot be addressed by index);
- the local array or structure is very large.

Compiling with `-Xptxas -v` shows each kernel's register usage and spilled bytes:

```text
ptxas info    : Used 128 registers, 64 bytes spill stores, 64 bytes spill loads
```

When there are spills, the usual remedies: hold less data per thread at once; unroll loops with `#pragma unroll` so local-array indices become compile-time constants; or tell the compiler the target configuration with `__launch_bounds__(maxThreads, minBlocks)` so it can trade off register allocation (which can, conversely, cause more spilling).

## Transfers between host and device {#主机与设备之间的传输}

CPU and GPU move data over PCIe, whose 4.0 x16 theoretical bandwidth is about 32 GB/s per direction and in practice 20-25 GB/s, **two orders of magnitude below device memory bandwidth**. So:

- move as little as possible between host and device, keeping data on the GPU and letting several kernels work on it in a row;
- one large transfer beats many small ones;
- use **pinned memory** (`cudaMallocHost`) for the host buffer, which is faster and is a prerequisite for asynchronous transfers; see [streams and concurrency](../tools/streams.md).

!!! interview "Answering in an interview"
    The order to answer a memory-optimization question in: global memory transfers in 32-byte sectors, so a warp reading 32 consecutive floats needs only 4 sectors while reading every other element doubles that, which is why you coalesce and prefer SoA; with alignment, `float4` cuts the number of memory instructions; shared memory has 32 banks, where different addresses in one bank conflict and the same address broadcasts, so declaring a two-dimensional array `[32][33]` removes the conflict by padding (or use a swizzle); check register spills with `-Xptxas -v`; and host-device copies should be few, large and from pinned memory.

## Exercises {#练习}

**1. Count the sectors.** A warp executes `float v = a[threadIdx.x * 3];` (with `a` aligned to 128 bytes and `threadIdx.x` from 0 to 31). How many 32-byte sectors does it touch? What fraction of the data is useful?

??? success "Answer"
    Thread k touches byte offset `12k`, from 0 to 372. At 32 bytes per sector the sectors are `floor(12k / 32)`, from 0 to 11, so **12 sectors**, 384 bytes transferred for 128 useful ones, **33%**.

**2. Find the bank conflict.** How many ways does the warp below conflict? How would you fix it?

```cuda
__shared__ double s[32 * 32];
double v = s[threadIdx.x * 2];
```

??? success "Answer"
    A `double` is 8 bytes, that is 2 words. For 64-bit accesses the hardware splits a warp's request into two phases of 16 threads each, and 32 banks hold exactly 16 doubles, so even a conflict-free access takes 2 transfers.

    Here thread k touches words `4k` and `4k+1`. Within one phase of 16 threads, threads k and k+8 land in the same bank (`4k % 32` equals `4(k+8) % 32`) at different addresses, a **2-way conflict**, so 4 transfers in all, twice the ideal.

    With `s[threadIdx.x]` instead, each phase's 16 threads touch 32 consecutive words covering all 32 banks, **conflict-free**. So the most direct fix is to have neighbouring threads touch neighbouring doubles; if the algorithm must stride, pad or swizzle.

**3. One-dimensional convolution.** Implement `y[i] = Σ_{k=-R}^{R} w[k+R] * x[i+k]` (treating out-of-range x as 0) with `R = 8`. Requirements: the weights go in constant memory; each block first loads the input it needs (including R halo elements on each side) into shared memory and then computes.

??? success "Answer"
    ```cuda title="conv1d.cu"
    // conv1d.cu - weights in constant memory, the input tile plus halo cached in shared memory
    // build: nvcc -O3 -arch=sm_75 conv1d.cu -o conv1d
    #include "common.cuh"

    constexpr int R = 8;
    constexpr int BLOCK = 256;
    __constant__ float c_w[2 * R + 1];

    __global__ void conv1d(const float* __restrict__ x, float* __restrict__ y, int n) {
      __shared__ float tile[BLOCK + 2 * R];
      const int base = blockIdx.x * BLOCK;
      // a cooperative load of BLOCK + 2R elements, so a thread may load more than one
      for (int j = threadIdx.x; j < BLOCK + 2 * R; j += blockDim.x) {
        int g = base + j - R;
        tile[j] = (g >= 0 && g < n) ? x[g] : 0.f;
      }
      __syncthreads();

      int i = base + threadIdx.x;
      if (i < n) {
        float acc = 0.f;
    #pragma unroll
        for (int k = 0; k <= 2 * R; ++k) acc += c_w[k] * tile[threadIdx.x + k];
        y[i] = acc;
      }
    }

    int main() {
      const int n = (1 << 22) + 123;
      std::vector<float> hx(n), hw(2 * R + 1), hy(n), ref(n);
      fill_random(hx, 1);
      fill_random(hw, 2);
      for (int i = 0; i < n; ++i) {
        float acc = 0.f;
        for (int k = -R; k <= R; ++k)
          if (i + k >= 0 && i + k < n) acc += hw[k + R] * hx[i + k];
        ref[i] = acc;
      }

      float *dx, *dy;
      CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpyToSymbol(c_w, hw.data(), hw.size() * sizeof(float)));

      const int blocks = (n + BLOCK - 1) / BLOCK;
      conv1d<<<blocks, BLOCK>>>(dx, dy, n);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
      bool ok = check_close(hy.data(), ref.data(), n, 1e-4f, 1e-5f);

      float ms = time_ms([&] { conv1d<<<blocks, BLOCK>>>(dx, dy, n); });
      std::printf("conv1d: %.3f ms, %.1f GB/s\n", ms, gbps(2.0 * n * sizeof(float), ms));
      CUDA_CHECK(cudaFree(dx));
      CUDA_CHECK(cudaFree(dy));
      return ok ? 0 : 1;
    }
    ```

    Each input element is used by 2R+1 outputs. Once it is in shared memory, each element is read from device memory only once (plus a little halo). Every thread of a warp reads the same weight at the same moment, which is exactly what constant memory's broadcast is for.

## Summary {#小结}

- [x] Global memory transfers in 32-byte sectors; keep a warp's addresses contiguous (coalescing) and prefer SoA.
- [x] With alignment, vector types like `float4` cut the number of memory instructions.
- [x] Shared memory has 32 banks; different addresses in one bank conflict while the same address broadcasts; remove conflicts with padding or a swizzle.
- [x] Constant memory suits small data that a whole warp reads uniformly.
- [x] Check register usage and spills with `-Xptxas -v`.
- [x] Host-device transfers are slow: do few of them, make them large, and use pinned memory.
