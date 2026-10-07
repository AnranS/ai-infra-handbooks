# The road to a fast GEMM

<p class="lead">Matrix multiplication (GEMM) is the most important kernel in deep learning, and almost all of an LLM's compute sits here. Getting from a naive version with one output element per thread to something close to cuBLAS takes coalescing, shared-memory tiling, register tiling, vectorization and double buffering in turn. This is the densest chapter of GPU optimization knowledge in the book, and the most common "write a kernel and optimize it" interview question.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How many floating-point operations are in an FP32 GEMM with M=N=K=4096? How little data can it move? Is it compute-bound or memory-bound?
    2. In the naive version, why is mapping `threadIdx.x` to the column so much faster than to the row?
    3. By how much does shared-memory tiling cut global-memory traffic?
    4. Why does register tiling (8×8 outputs per thread) win several times more again? Where does the bottleneck move from and to?
    5. What problem does double buffering solve?

??? success "Answers (try it yourself first, then expand)"
    1. $2 \times 4096^3 \approx 1.37 \times 10^{11}$ operations; at minimum it reads A and B and writes C, $4096^2$ floats each, about 200 MB. That is an arithmetic intensity of about 680 FLOP/byte, far above the ridge point, so it is compute-bound in theory.
    2. With `threadIdx.x` as the column, a warp reads 32 consecutive elements of one row of B (coalesced), reads one element of A (broadcast), and writes C contiguously; as the row, each thread reads a different row of B, with no coalescing at all.
    3. Each element read from global memory into shared memory is used TILE times (TILE = 32 in this chapter), so global traffic falls to 1/32.
    4. Each thread computes 8×8 outputs in registers: each step reads 8 + 8 values from shared memory and does 64 multiply-adds, lifting the reuse from the shared-memory level to the register level (8 times the ratio of shared-memory tiling alone). The bottleneck moves from shared-memory bandwidth and memory instructions to the computation itself.
    5. Loading the next tile from global memory while computing on the current one: two shared-memory buffers alternate so the load latency hides behind the compute, rather than running "load, synchronize, compute, synchronize" in series.

A six-panel strip before the text:

<!-- comic ../assets/comics/gemm.webp is in Chinese; put it back once the English version exists -->

## The problem and the ceiling {#问题定义与性能上限}

Compute $C = \alpha AB + \beta C$, where A is M×K, B is K×N and C is M×N, all row-major FP32.

- **Compute**: each output element takes K multiply-adds, so $2MNK$ floating-point operations. At M=N=K=4096 that is about 137 billion.
- **Minimum traffic**: read A and B, read and write C, about $4(MK + KN + 2MN)$ bytes. At 4096 that is about 268 MB.
- **Arithmetic intensity**: about 1374e9 / 268e6 ≈ 512 FLOP/byte, far above any GPU's ridge point (about 9.6 for A100 FP32).

So **in theory** a large GEMM is compute-bound, with a ceiling of peak FP32 throughput (about 19.5 TFLOPS on an A100, giving about 7 ms for a 4096 GEMM). But the "minimum traffic" assumes each value is read from memory once, which the naive version comes nowhere near: each element of A is read N times and each of B M times. **The through-line of GEMM optimization is raising the reuse step by step until the real arithmetic intensity is above the ridge point, and then optimizing the computation itself.**

This chapter uses an FP32 CUDA Core implementation to make that line clear. Real LLM GEMMs run on FP16/BF16/FP8 Tensor Cores, which follow exactly the same thinking with a matrix instruction at the innermost level; see [Tensor Cores](../advanced/tensor-core.md).

## v0 to v1: the naive version and coalescing {#v0--v1朴素实现与合并访问}

Each thread computes one element of C:

```cuda
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;   // v0: x is the row
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}
```

A warp's 32 threads have consecutive `threadIdx.x`, which here means 32 different **rows**: reading A has 32 threads touching the same column of 32 different rows, K elements apart, with no coalescing at all; reading B has all 32 reading the same element (a broadcast).

**v1** changes one thing: `threadIdx.x` becomes the **column**. Now reading B has 32 threads touching 32 consecutive elements of one row, coalesced; reading A has them all read one element, broadcast; and writing C is contiguous too. Swapping the index mapping alone is worth several times the performance. **This is the most basic habit in a two-dimensional problem: let `threadIdx.x` run along the contiguous dimension.**

But v1 is still slow: each multiply-add (2 FLOP) reads two floats (8 bytes) from global memory (most of it hitting L1/L2 in practice), an arithmetic intensity of only 0.25 FLOP/byte.

## v2: shared-memory tiling {#v2共享内存分块}

Cut C into 32×32 tiles, one per block. Computing a tile needs 32 rows of A and 32 columns of B, sliced along K into 32×32 pieces handled one at a time:

![Figure: shared-memory tiling](../assets/figures/tiled-gemm.svg){.aig-svg}

1. the block's 1024 threads **cooperate** to load one 32×32 tile of A and one of B into shared memory (one element each, with coalesced reads);
2. `__syncthreads()`;
3. each thread does 32 multiply-adds from shared memory, accumulating in its own register;
4. `__syncthreads()` (so nobody overwrites a tile others are still reading);
5. move to the next slice along K and repeat.

```cuda
for (int k0 = 0; k0 < K; k0 += TILE) {
  As[ty][tx] = A[row * K + k0 + tx];
  Bs[ty][tx] = B[(k0 + ty) * N + col];
  __syncthreads();
  for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
  __syncthreads();
}
```

Each element loaded from global memory is now used 32 times in shared memory, so **global traffic falls to 1/32**.

Even so, the inner loop still reads shared memory twice per multiply-add (`As[ty][k]` broadcasts within the warp and `Bs[k][tx]` is contiguous, so neither conflicts). Shared-memory bandwidth is far higher than device memory's but still finite: the bottleneck has moved from device memory to **shared-memory bandwidth** and to the sheer number of memory instructions.

## v3: one-dimensional register tiling {#v3一维寄存器分块}

Carry the reuse idea one level further: have each thread compute **several** outputs, lifting the reuse from shared memory to **registers**.

v3 gives each thread TM = 8 elements of one column. A block owns a 64×64 tile of C and takes 8 along K at a time:

```cuda
for (int dot = 0; dot < BK; ++dot) {
  float b = Bs[dot * BN + threadCol];                 // one read of B
  for (int r = 0; r < TM; ++r)
    acc[r] += As[(threadRow * TM + r) * BK + dot] * b;  // reused 8 times
}
```

Each B element read out of shared memory sits in a register and is reused 8 times. The shared-memory reads per output fall from 2 to about 1.1.

## v4: two-dimensional register tiling {#v4二维寄存器分块}

Further still: each thread computes a TM×TN = 8×8 tile of the output. At each `dot` step a thread reads 8 elements of A and 8 of B from shared memory into registers and does **8×8 = 64 multiply-adds (an outer product)**:

```cuda
for (int dot = 0; dot < BK; ++dot) {
  for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dot];
  for (int i = 0; i < TN; ++i) regN[i] = Bs[dot * BN + threadCol * TN + i];
  for (int m = 0; m < TM; ++m)
    for (int n = 0; n < TN; ++n) acc[m][n] += regM[m] * regN[n];
}
```

16 shared-memory reads now support 64 multiply-adds, 8 times v2's ratio. Each block now owns a 128×128 tile of C with 256 threads, each holding 64 registers of accumulators.

This is **the step that pays most** in GEMM optimization. From here the inner loop is mostly FFMA instructions and performance starts approaching the CUDA Core ceiling.

!!! tip "Why register tiling works so well"
    Every level of tiling does the same thing: an a×b output tile needs a elements of A and b of B for a×b multiply-adds. **Traffic grows with the perimeter and compute with the area.** The larger and squarer the tile, the more compute each access supports. Every level of a GPU's storage does this: device memory to shared memory (block-level tiling), shared memory to registers (thread-level tiling), and the Tensor Core version adds a warp-level tile in between.

## v5: vectorized loads {#v5向量化访存}

In v4's inner loop, the 8 elements of `As` are BK apart, so they are 8 independent scalar loads. v5 changes two things:

1. **store A transposed in shared memory**: `As` becomes `[BK][BM]`, so a thread's 8 A elements are contiguous and can be read with two 128-bit instructions;
2. **use `float4` for all global reads and writes**: a quarter as many instructions for reading A and B and writing C.

These cut instructions and address arithmetic, raising FFMA's share of the instruction stream.

## The complete code {#完整代码}

```cuda title="gemm.cu"
// gemm.cu - SGEMM from naive to two-dimensional register tiling plus vectorization
// build: nvcc -O3 -arch=sm_80 gemm.cu -o gemm
// against cuBLAS: nvcc -O3 -arch=sm_80 -DWITH_CUBLAS gemm.cu -o gemm -lcublas
// requires M, N and K to be multiples of 128 (bounds are not handled, to keep the code short)
#include "common.cuh"
#ifdef WITH_CUBLAS
#include <cublas_v2.h>
#endif

// ---------------- v0: threadIdx.x is the row, so nothing coalesces ----------------
__global__ void sgemm_v0(int M, int N, int K, float alpha, const float* __restrict__ A,
                         const float* __restrict__ B, float beta, float* __restrict__ C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// ---------------- v1: threadIdx.x is the column, so reading B and writing C coalesce ----------------
__global__ void sgemm_v1(int M, int N, int K, float alpha, const float* __restrict__ A,
                         const float* __restrict__ B, float beta, float* __restrict__ C) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// ---------------- v2: 32x32 shared-memory tiling ----------------
template <int TILE>
__global__ void sgemm_v2(int M, int N, int K, float alpha, const float* __restrict__ A,
                         const float* __restrict__ B, float beta, float* __restrict__ C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    As[ty][tx] = A[row * K + k0 + tx];
    Bs[ty][tx] = B[(k0 + ty) * N + col];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// ---------------- v3: one-dimensional register tiling, TM elements of one column per thread ----------------
template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM)
sgemm_v3(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
         float beta, float* __restrict__ C) {
  static_assert(BM * BK == (BM * BN) / TM && BN * BK == (BM * BN) / TM, "one element per thread per load");
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  const int threadCol = threadIdx.x % BN, threadRow = threadIdx.x / BN;
  const int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  float acc[TM] = {0.f};
  for (int k0 = 0; k0 < K; k0 += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
      const float b = Bs[dot * BN + threadCol];
#pragma unroll
      for (int r = 0; r < TM; ++r) acc[r] += As[(threadRow * TM + r) * BK + dot] * b;
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < TM; ++r) {
    float& c = C[(threadRow * TM + r) * N + threadCol];
    c = alpha * acc[r] + beta * c;
  }
}

// ---------------- v4: two-dimensional register tiling, TM x TN per thread ----------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_v4(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
         float beta, float* __restrict__ C) {
  constexpr int kThreads = (BM * BN) / (TM * TN);
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  const int threadCol = threadIdx.x % (BN / TN), threadRow = threadIdx.x / (BN / TN);
  const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
  constexpr int strideA = kThreads / BK;
  const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
  constexpr int strideB = kThreads / BN;
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  float acc[TM][TN] = {{0.f}};
  float regM[TM], regN[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int off = 0; off < BM; off += strideA)
      As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * K + innerColA];
#pragma unroll
    for (int off = 0; off < BK; off += strideB)
      Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dot];
#pragma unroll
      for (int i = 0; i < TN; ++i) regN[i] = Bs[dot * BN + threadCol * TN + i];
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) acc[m][n] += regM[m] * regN[n];
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      float& c = C[(threadRow * TM + m) * N + threadCol * TN + n];
      c = alpha * acc[m][n] + beta * c;
    }
}

// ---------------- v5: v4 plus A stored transposed in shared memory plus float4 accesses ----------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_v5(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
         float beta, float* __restrict__ C) {
  constexpr int kThreads = (BM * BN) / (TM * TN);
  static_assert(BM * BK / 4 == kThreads && BK * BN / 4 == kThreads, "one float4 per thread per load");
  __shared__ __align__(16) float As[BK * BM];   // stored transposed as As[k][m]; a float4 access needs 16-byte alignment
  __shared__ __align__(16) float Bs[BK * BN];
  const int threadCol = threadIdx.x % (BN / TN), threadRow = threadIdx.x / (BN / TN);
  const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
  const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  float acc[TM][TN] = {{0.f}};
  float regM[TM], regN[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    const float4 a4 = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a4.w;
    reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();
    A += BK;
    B += BK * N;
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; i += 4)
        *reinterpret_cast<float4*>(&regM[i]) = *reinterpret_cast<const float4*>(&As[dot * BM + threadRow * TM + i]);
#pragma unroll
      for (int i = 0; i < TN; i += 4)
        *reinterpret_cast<float4*>(&regN[i]) = *reinterpret_cast<const float4*>(&Bs[dot * BN + threadCol * TN + i]);
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) acc[m][n] += regM[m] * regN[n];
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      float4* cp = reinterpret_cast<float4*>(&C[(threadRow * TM + m) * N + threadCol * TN + n]);
      float4 c = *cp;
      c.x = alpha * acc[m][n + 0] + beta * c.x;
      c.y = alpha * acc[m][n + 1] + beta * c.y;
      c.z = alpha * acc[m][n + 2] + beta * c.z;
      c.w = alpha * acc[m][n + 3] + beta * c.w;
      *cp = c;
    }
}

// ---------------- launch wrappers ----------------
using Launcher = void (*)(int, int, int, float, const float*, const float*, float, float*);

void launch_v0(int M, int N, int K, float a, const float* A, const float* B, float b, float* C) {
  dim3 block(32, 32), grid(M / 32, N / 32);
  sgemm_v0<<<grid, block>>>(M, N, K, a, A, B, b, C);
}
void launch_v1(int M, int N, int K, float a, const float* A, const float* B, float b, float* C) {
  dim3 block(32, 32), grid(N / 32, M / 32);
  sgemm_v1<<<grid, block>>>(M, N, K, a, A, B, b, C);
}
void launch_v2(int M, int N, int K, float a, const float* A, const float* B, float b, float* C) {
  dim3 block(32, 32), grid(N / 32, M / 32);
  sgemm_v2<32><<<grid, block>>>(M, N, K, a, A, B, b, C);
}
void launch_v3(int M, int N, int K, float a, const float* A, const float* B, float b, float* C) {
  constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
  dim3 grid(N / BN, M / BM);
  sgemm_v3<BM, BN, BK, TM><<<grid, (BM * BN) / TM>>>(M, N, K, a, A, B, b, C);
}
void launch_v4(int M, int N, int K, float a, const float* A, const float* B, float b, float* C) {
  constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  dim3 grid(N / BN, M / BM);
  sgemm_v4<BM, BN, BK, TM, TN><<<grid, (BM * BN) / (TM * TN)>>>(M, N, K, a, A, B, b, C);
}
void launch_v5(int M, int N, int K, float a, const float* A, const float* B, float b, float* C) {
  constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  dim3 grid(N / BN, M / BM);
  sgemm_v5<BM, BN, BK, TM, TN><<<grid, (BM * BN) / (TM * TN)>>>(M, N, K, a, A, B, b, C);
}

// the CPU reference: an i-k-j loop order, which is cache-friendly
void gemm_cpu(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
  std::vector<float> acc(N);
  for (int i = 0; i < M; ++i) {
    std::fill(acc.begin(), acc.end(), 0.f);
    for (int k = 0; k < K; ++k) {
      const float a = A[i * K + k];
      const float* b = B + k * N;
      for (int j = 0; j < N; ++j) acc[j] += a * b[j];
    }
    for (int j = 0; j < N; ++j) C[i * N + j] = alpha * acc[j] + beta * C[i * N + j];
  }
}

int main(int argc, char** argv) {
  const int check_n = 512;                               // correctness is checked on a small matrix (which the CPU handles quickly)
  const int bench_n = argc > 1 ? std::atoi(argv[1]) : 4096;
  const float alpha = 1.f, beta = 0.5f;

  struct { const char* name; Launcher fn; } versions[] = {
      {"v0 naive (x->row)", launch_v0},    {"v1 coalesced (x->col)", launch_v1},
      {"v2 smem tiling", launch_v2},        {"v3 1D register tiling", launch_v3},
      {"v4 2D register tiling", launch_v4}, {"v5 vectorized", launch_v5},
  };

  // ---- correctness ----
  {
    const int n = check_n;
    std::vector<float> hA(n * n), hB(n * n), hC(n * n), ref(n * n), got(n * n);
    fill_random(hA, 1);
    fill_random(hB, 2);
    fill_random(hC, 3);
    ref = hC;
    gemm_cpu(n, n, n, alpha, hA.data(), hB.data(), beta, ref.data());
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    bool ok = true;
    for (auto& v : versions) {
      CUDA_CHECK(cudaMemcpy(C, hC.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
      v.fn(n, n, n, alpha, A, B, beta, C);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaMemcpy(got.data(), C, n * n * sizeof(float), cudaMemcpyDeviceToHost));
      std::printf("%-24s ", v.name);
      ok &= check_close(got.data(), ref.data(), n * n, 1e-4f, 1e-4f);
    }
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    if (!ok) return 1;
  }

  // ---- performance ----
  const int n = bench_n;
  const size_t elems = static_cast<size_t>(n) * n;
  std::vector<float> h(elems);
  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&B, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&C, elems * sizeof(float)));
  fill_random(h, 4);
  CUDA_CHECK(cudaMemcpy(A, h.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, h.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(C, 0, elems * sizeof(float)));
  const double flops = 2.0 * n * n * n;
  std::printf("\nM = N = K = %d\n%-24s %10s %10s\n", n, "version", "time(ms)", "TFLOPS");

#ifdef WITH_CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
  // cuBLAS is column-major: row-major C = A B is column-major C^T = B^T A^T
  float ms_cublas = time_ms([&] {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n, A, n, &beta, C, n);
  }, 10, 2);
  std::printf("%-24s %10.3f %10.2f\n", "cuBLAS", ms_cublas, tflops(flops, ms_cublas));
  cublasDestroy(handle);
#endif
  for (auto& v : versions) {
    const int iters = (v.fn == launch_v0) ? 1 : 10;   // v0 is too slow to run more than once
    float ms = time_ms([&] { v.fn(n, n, n, alpha, A, B, beta, C); }, iters, 1);
    std::printf("%-24s %10.3f %10.2f\n", v.name, ms, tflops(flops, ms));
  }
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
```

Record what you get. Based on published experience (Simon Boehm's *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*, which goes from naive to above 90% of cuBLAS on an A6000), you should see roughly this trend: v0 to v1 several times, v2 about double again, v4 several times over v2, and v5 closing to 70-80% of cuBLAS. The exact ratios vary by GPU, so **trust your own measurements**.

| Version | Key change | Time | TFLOPS | % of cuBLAS |
| --- | --- | --- | --- | --- |
| cuBLAS | baseline | | | 100% |
| v0 | naive | | | |
| v1 | coalescing | | | |
| v2 | shared-memory tiling | | | |
| v3 | one-dimensional register tiling | | | |
| v4 | two-dimensional register tiling | | | |
| v5 | vectorization | | | |

## Where to go next {#接下来的优化方向}

Up to v5, several classic optimizations are still missing, and they are what closes the last of the gap to cuBLAS:

**Warp tiling.** Add a level between the block tile and the thread tile: each warp owns a sub-tile of the block (64×32, say). That concentrates the shared memory a warp touches, cuts bank conflicts, and prepares the way for Tensor Cores (whose mma instructions are per warp). CUTLASS's hierarchy is exactly block tile, warp tile, thread tile / mma instruction.

**Double buffering and multi-stage pipelines.** In v5's loop, "load the next tile" and "compute on the current one" run in series: the math units idle during the load and the memory units during the compute. Double buffering allocates two shared-memory buffers and prefetches tile i+1 while computing tile i, hiding the load latency behind the compute. From Ampere on, `cp.async` copies from global memory into shared memory asynchronously without going through registers, giving 3-4 pipeline stages; see [asynchronous copies](../advanced/async-hopper.md).

**Removing shared-memory bank conflicts.** v5 has a few conflicts when writing A transposed into shared memory, which padding or a swizzle removes.

**Autotuning.** The best BM, BN, BK, TM and TN depend on both the GPU and the matrix shape. Real libraries pick different configurations for different shapes. When M is small (a small batch during LLM decode, say), strategies like split-K are needed to fill the GPU.

**Tensor Cores.** FP32 CUDA Cores cap out at 19.5 TFLOPS on an A100 while BF16 Tensor Cores reach 312, a factor of 16. **Every real deep-learning GEMM runs on Tensor Cores**; see [Tensor Cores](../advanced/tensor-core.md). GPUs from A100 on also support TF32: the inputs stay FP32 while the Tensor Core computes with a 10-bit mantissa, enabled in PyTorch with `torch.backends.cuda.matmul.allow_tf32 = True`.

## Answering in an interview {#面试怎么答}

"Write a matrix multiply and optimize it" is one of the most frequent questions. A suggested rhythm:

1. write v1 (with `threadIdx.x` as the column), compute its arithmetic intensity, and point out that it is memory-bound;
2. write v2's shared-memory tiling (getting both `__syncthreads()` in the right places and explaining why), noting that traffic falls to 1/TILE;
3. explain register tiling and the "perimeter against area" analogy out loud, ideally writing v4's inner loop;
4. go on to list double buffering, warp tiling, vectorization, Tensor Cores and CUTLASS, saying what each solves;
5. finish with numbers you measured yourself, such as "my 4096 SGEMM reaches X% of cuBLAS on an A100".

!!! interview "Answering in an interview"
    GEMM optimization is the most common "walk me through how you optimized something" question: a 4096³ FP32 GEMM is about 137 GFLOP against a minimum of about 200 MB of traffic, so it is compute-bound in theory and the through-line is raising reuse level by level. In the naive version, first map `threadIdx.x` to the contiguous dimension; shared-memory tiling cuts global traffic to 1/tile; two-dimensional register tiling (8×8 outputs per thread) is the step that pays most and moves the bottleneck from memory to compute; after that come vectorization, double buffering or a `cp.async` pipeline, warp tiling, removing bank conflicts, and tuning. Finish with Tensor Cores: the same thinking with a matrix instruction innermost.

## Exercises {#练习}

**1. A calculation.** In v2 (TILE=32) and v4 (BM=BN=128, TM=TN=8), how many shared-memory reads does an average multiply-add take? How much does each block load from global memory in total (for M=N=K=4096)?

??? success "Answer"
    - **Shared-memory reads**: v2 reads `As` and `Bs` once each per multiply-add, so **2**. v4 reads 8 + 8 = 16 per `dot` step for 64 multiply-adds, an average of **0.25**.
    - **Global loads**: v2's block owns a 32×32 tile of C and needs 32 rows of A (32×4096) and 32 columns of B (4096×32), so 2 × 32 × 4096 × 4 B = 1 MB; with (4096/32)² = 16384 blocks that is 16 GB in total. v4's block owns 128×128 and loads 2 × 128 × 4096 × 4 B = 4 MB; with 1024 blocks that is 4 GB. Four times the tile edge, a quarter of the global traffic. (In practice much of it hits L2, so the device-memory traffic is lower still.)

**2. Supporting any size.** Add bounds handling to v2 so it computes a GEMM with M=1000, N=999, K=777 correctly.

??? success "Approach"
    Fill zeros where the load is out of range: `As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;`, and likewise check `k0 + ty < K && col < N` for `Bs`. The computation itself needs no change, since the zeros do not affect the result. Check `row < M && col < N` when writing back. **Note that out-of-range threads must not return early**: they still have to take part in the loads and the `__syncthreads()`. Bounds handling in v4 and v5 is more involved, and real libraries usually split "full tiles" and "edge tiles" into separate code paths, or pad the input.

**3. Implement double buffering.** On top of v4, allocate two copies of `As` and `Bs`, issue the global read for the next tile (into registers) before computing the current one, and write it into the other shared-memory copy afterwards. Compare the performance.

??? success "Approach"
    The structure (in pseudocode):

    ```cuda
    load_tile_to_regs(0);  store_regs_to_smem(buf = 0);  __syncthreads();
    for (int t = 0; t < numTiles; ++t) {
      if (t + 1 < numTiles) load_tile_to_regs(t + 1);   // issue the global read without waiting for it
      compute_from_smem(buf);                            // overlaps with the read above
      if (t + 1 < numTiles) store_regs_to_smem(buf ^ 1);
      __syncthreads();
      buf ^= 1;
    }
    ```

    Because the reads and the writes go to different buffers, one `__syncthreads()` per iteration is enough. The cost is more registers and shared memory and possibly lower occupancy, so measure the trade-off. On sm_80 and above, `cp.async` makes this both simpler and faster.

## Summary {#小结}

- [x] GEMM is compute-bound in theory, and the through-line of optimization is raising reuse level by level: shared-memory tiling, then register tiling.
- [x] Map `threadIdx.x` to the contiguous dimension; both `__syncthreads()` in the shared-memory tiling loop are indispensable.
- [x] The larger and squarer the tile, the lower the traffic-to-compute ratio; two-dimensional register tiling is the step that pays most.
- [x] Further: vectorization, double buffering or a cp.async pipeline, warp tiling, removing bank conflicts, tuning.
- [x] Deep-learning GEMMs run on Tensor Cores, with the same thinking and a matrix instruction innermost.
