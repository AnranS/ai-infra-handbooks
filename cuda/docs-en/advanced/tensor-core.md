# Tensor Cores

<p class="lead">A Tensor Core is hardware dedicated to small matrix multiply-accumulates, with more than ten times the throughput of ordinary CUDA Cores, and essentially all modern LLM training and inference runs on it. This chapter covers how it works, its several programming interfaces (WMMA, mma.sync, wgmma), and what you have to know about data layout and precision.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What does one Tensor Core instruction compute? At what granularity (thread, warp, block) is it issued?
    2. What are the peak FP32 CUDA Core and BF16 Tensor Core numbers on an A100?
    3. How is a WMMA fragment's data spread across the threads? Does the programmer need to care?
    4. How large are A, B and C in `mma.sync.m16n8k16`?
    5. How do FP16 and BF16 differ? Why is the accumulator usually FP32?

??? success "Answers (try it yourself first, then expand)"
    1. One small matrix multiply-accumulate $D = A \times B + C$ (16×8×16, say), issued cooperatively by a warp (on Hopper, by a warpgroup of 4 warps), with the data living in the threads' registers.
    2. About 19.5 TFLOPS of FP32 on CUDA Cores; 312 TFLOPS of dense BF16 on Tensor Cores, an order of magnitude more.
    3. The distribution is implementation-defined and undocumented: which elements of a fragment each thread holds is unspecified, and you can only load, compute and store it as a whole, never relying on the layout (use mma.sync's explicit layout when you need to work element by element).
    4. A is 16×16, B is 16×8, and C and D are 16×8 (m16n8k16).
    5. FP16 has 5 exponent bits and 10 mantissa bits, BF16 has 8 and 7 (the same range as FP32, so it rarely overflows). The accumulator is FP32 because a long accumulation of thousands of terms accumulates error quickly in low precision.

A six-panel strip before the text:

<!-- comic ../assets/comics/tensor-core.webp is in Chinese; put it back once the English version exists -->

## What a Tensor Core does {#tensor-core-做什么}

A Tensor Core computes **D = A × B + C**, where A, B, C and D are small matrices (16×16, say). Unlike a CUDA Core instruction doing one scalar multiply-add, one Tensor Core instruction is **issued cooperatively by a whole warp**, completes a small matrix multiply, and keeps the data spread across the warp's 32 threads' registers.

The throughput (NVIDIA's official dense figures):

| GPU | FP32 CUDA Core | TF32 Tensor | BF16/FP16 Tensor | FP8 Tensor |
| --- | --- | --- | --- | --- |
| A100 | 19.5 TFLOPS | 156 TFLOPS | 312 TFLOPS | — |
| H100 SXM | 67 TFLOPS | 495 TFLOPS | 989 TFLOPS | 1979 TFLOPS |

That is more than an order of magnitude. So any GEMM-dominated computation should run on Tensor Cores whenever the precision allows.

The data types each generation added:

| Architecture | New Tensor Core types |
| --- | --- |
| Volta sm_70 | FP16 |
| Turing sm_75 | INT8, INT4 |
| Ampere sm_80 | BF16, TF32, FP64 |
| Ada sm_89 / Hopper sm_90 | FP8 (E4M3, E5M2) |
| Blackwell sm_100 | FP6, FP4 (including block-scaled formats such as NVFP4 and MXFP4) |

## Precision: FP16, BF16, TF32, FP8 {#精度fp16bf16tf32fp8}

| Format | sign/exponent/mantissa | Character |
| --- | --- | --- |
| FP32 | 1/8/23 | the baseline |
| TF32 | 1/8/10 | not a storage format but the precision a Tensor Core computes FP32 inputs at |
| FP16 | 1/5/10 | better precision but a small range (about 65504 at most), needing loss scaling in training |
| BF16 | 1/8/7 | the same range as FP32 at lower precision; the mainstream format for LLM training and inference |
| FP8 E4M3 | 1/4/3 | common for inference and the forward pass, needing scaling factors |
| FP8 E5M2 | 1/5/2 | a larger range, common for gradients in training |

A matrix multiply accumulates K products, and low-precision accumulation builds error quickly as K grows, so **the accumulator is generally FP32** with low-precision inputs and outputs. The Tensor Core instructions support "FP16/BF16 in, FP32 accumulate" natively. An FP8 GEMM also needs scaling factors (per tensor, per row or per block) to map the values into FP8's range; DeepSeek-V3's FP8 training used fine-grained block scaling.

## Three generations of interface {#三代编程接口}

| Interface | Architecture | Granularity | Character |
| --- | --- | --- | --- |
| **WMMA** (`nvcuda::wmma`) | sm_70+ | warp, 16×16×16 and so on | a simple C++ API; the fragment layout is opaque |
| **mma.sync** (PTX) | the mainstream from sm_80 | warp, m16n8k16 and so on | an explicit register layout, used with `ldmatrix`, the best performance; what high-performance libraries do on Ampere |
| **wgmma** (PTX) | sm_90a | warpgroup (4 warps), m64nNk16 | asynchronous, with operands straight from shared memory; required for full speed on Hopper |
| **tcgen05** (PTX) | sm_100a | issued by one thread, up to two SMs cooperating | the result lives in dedicated tensor memory; Blackwell only |

In real work a high-performance GEMM rarely writes PTX by hand and instead uses **CUTLASS / CuTe** (a C++ template library) or **Triton**, which wrap these instructions along with the data movement they need. But understanding how WMMA and mma.sync work is what lets you read those libraries and customize them.

First draw the three levels of tiling in a high-performance GEMM: a block owns a tile of C and accumulates along K piece by piece, the block splits by warp, and each mma instruction within a warp computes a small tile. Drag to rotate and change the tile sizes to see what happens to shared memory and arithmetic intensity:

<div class="aig-widget" data-widget="gemm3d"></div>

## WMMA: the simplest Tensor Core programming {#wmma最简单的-tensor-core-编程}

WMMA splits a matrix into 16×16 **fragments**. A fragment is an object "spread across a warp's 32 threads", and you need not know which elements each thread holds; the whole warp simply calls these functions together:

```cuda
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

wmma::fill_fragment(c, 0.f);
wmma::load_matrix_sync(a, ptr_a, lda);     // load a 16x16 from memory (global or shared), with lda as the row stride
wmma::load_matrix_sync(b, ptr_b, ldb);
wmma::mma_sync(c, a, b, c);                // c = a * b + c
wmma::store_matrix_sync(ptr_c, c, ldc, wmma::mem_row_major);
```

The requirements: pointers aligned to 32 bytes, and `ldm` a multiple of 8 for half (16 bytes).

The program below implements a GEMM with two FP16 inputs and FP32 accumulation:

- **v1**: each warp owns one 16×16 tile of C and loads its fragments straight from global memory;
- **v2**: each block (4 warps) owns a 64×64 tile of C, first moving tiles of A and B into shared memory with 128-bit vectorized reads (padding each row by 8 halves to cut bank conflicts), after which each warp loads its fragments from shared memory and owns 32×32 of the output (2×2 fragments). One A fragment is reused by 2 B fragments and vice versa.

```cuda title="wmma_gemm.cu"
// wmma_gemm.cu - Tensor Cores through WMMA: FP16 in, FP32 accumulate
// build: nvcc -O3 -arch=sm_75 wmma_gemm.cu -o wmma_gemm
// requires M and N to be multiples of 64 and K a multiple of 32
#include "common.cuh"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// v1: one warp computes a 16x16 tile of C, loading its fragments straight from global memory
__global__ void wmma_v1(int M, int N, int K, const half* __restrict__ A, const half* __restrict__ B,
                        float* __restrict__ C) {
  const int warp = threadIdx.x / 32;
  const int tile_m = blockIdx.y;                  // which group of 16 rows
  const int tile_n = blockIdx.x * 4 + warp;       // 4 warps per block, laid out across
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.f);
  for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a, A + tile_m * 16 * K + k, K);
    wmma::load_matrix_sync(b, B + k * N + tile_n * 16, N);
    wmma::mma_sync(acc, a, b, acc);
  }
  wmma::store_matrix_sync(C + tile_m * 16 * N + tile_n * 16, acc, N, wmma::mem_row_major);
}

// v2: a block owns 64x64 with its 4 warps in a 2x2 arrangement owning 32x32 each; A and B go through shared memory
constexpr int BM = 64, BN = 64, BK = 32, PAD = 8;
__global__ void __launch_bounds__(128) wmma_v2(int M, int N, int K, const half* __restrict__ A,
                                               const half* __restrict__ B, float* __restrict__ C) {
  __shared__ __align__(32) half As[BM][BK + PAD];
  __shared__ __align__(32) half Bs[BK][BN + PAD];
  const int tid = threadIdx.x, warp = tid / 32;
  const int warp_m = warp / 2, warp_n = warp % 2;
  const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[2];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    // a 64x32 tile of A is 256 uint4 (8 halves each), so each of the 128 threads moves 2
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      int idx = tid + t * 128, r = idx / 4, c = (idx % 4) * 8;
      *reinterpret_cast<uint4*>(&As[r][c]) = *reinterpret_cast<const uint4*>(&A[(row0 + r) * K + k0 + c]);
    }
    // a 32x64 tile of B is 256 uint4 likewise
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      int idx = tid + t * 128, r = idx / 8, c = (idx % 8) * 8;
      *reinterpret_cast<uint4*>(&Bs[r][c]) = *reinterpret_cast<const uint4*>(&B[(k0 + r) * N + col0 + c]);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[warp_m * 32 + i * 16][kk], BK + PAD);
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[kk][warp_n * 32 + j * 16], BN + PAD);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float* dst = C + (row0 + warp_m * 32 + i * 16) * N + col0 + warp_n * 32 + j * 16;
      wmma::store_matrix_sync(dst, acc[i][j], N, wmma::mem_row_major);
    }
}

int main(int argc, char** argv) {
  require_sm(7, 0);
  const int M = argc > 1 ? std::atoi(argv[1]) : 2048, N = M, K = M;
  std::vector<float> fa(static_cast<size_t>(M) * K), fb(static_cast<size_t>(K) * N);
  fill_random(fa, 1);
  fill_random(fb, 2);
  std::vector<half> ha(fa.size()), hb(fb.size());
  for (size_t i = 0; i < fa.size(); ++i) { ha[i] = __float2half(fa[i]); fa[i] = __half2float(ha[i]); }
  for (size_t i = 0; i < fb.size(); ++i) { hb[i] = __float2half(fb[i]); fb[i] = __half2float(hb[i]); }

  // the CPU reference checks only the first 64 rows, to keep it quick
  const int check_rows = 64;
  std::vector<float> ref(static_cast<size_t>(check_rows) * N, 0.f);
  for (int i = 0; i < check_rows; ++i)
    for (int k = 0; k < K; ++k) {
      float a = fa[static_cast<size_t>(i) * K + k];
      for (int j = 0; j < N; ++j) ref[static_cast<size_t>(i) * N + j] += a * fb[static_cast<size_t>(k) * N + j];
    }

  half *dA, *dB;
  float* dC;
  CUDA_CHECK(cudaMalloc(&dA, ha.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB, hb.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dC, static_cast<size_t>(M) * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, ha.data(), ha.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hb.data(), hb.size() * sizeof(half), cudaMemcpyHostToDevice));

  std::vector<float> got(static_cast<size_t>(check_rows) * N);
  const double flops = 2.0 * M * N * K;
  bool ok = true;
  auto bench = [&](const char* name, auto launch) {
    CUDA_CHECK(cudaMemset(dC, 0, static_cast<size_t>(M) * N * sizeof(float)));
    launch();
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("%-8s ", name);
    ok &= check_close(got.data(), ref.data(), got.size(), 1e-3f, 1e-2f);
    float ms = time_ms(launch);
    std::printf("%-8s %.3f ms, %.2f TFLOPS\n", "", ms, tflops(flops, ms));
  };
  bench("wmma v1", [&] { wmma_v1<<<dim3(N / 64, M / 16), 128>>>(M, N, K, dA, dB, dC); });
  bench("wmma v2", [&] { wmma_v2<<<dim3(N / BN, M / BM), 128>>>(M, N, K, dA, dB, dC); });
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return ok ? 0 : 1;
}
```

v2 is still far from cuBLAS: no double-buffered pipeline, warp tiles that are too small, and a shared-memory layout with no swizzle. Those are precisely what CUTLASS does. But it already shows the complete hierarchy of a Tensor Core GEMM: **global memory, shared memory (block tiling), fragments/registers (warp tiling), the matrix instruction**.

## mma.sync: the register layout is explicit {#mmasync寄存器布局是明确的}

WMMA's fragment layout is undocumented, which makes it hard to combine with other optimizations (computing elementwise on the result in registers, or feeding one GEMM's output straight into the next, which is exactly what FlashAttention needs). PTX's `mma.sync` specifies which elements each thread holds. Take the most common `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (A is 16×16, B is 16×8, C/D are 16×8):

With `g = lane / 4` (the group, 0-7) and `t = lane % 4` (the position within it, 0-3):

| Operand | Registers per thread | Element positions (row, column) |
| --- | --- | --- |
| A (16×16, FP16) | 4 32-bit registers of 2 halves each | a0,a1: (g, 2t), (g, 2t+1); a2,a3: (g+8, 2t), (g+8, 2t+1); a4,a5: (g, 2t+8), (g, 2t+9); a6,a7: (g+8, 2t+8), (g+8, 2t+9) |
| B (16×8, FP16) | 2 32-bit registers | b0,b1: (2t, g), (2t+1, g); b2,b3: (2t+8, g), (2t+9, g) |
| C/D (16×8, FP32) | 4 floats | c0,c1: (g, 2t), (g, 2t+1); c2,c3: (g+8, 2t), (g+8, 2t+1) |

The program below has one warp compute a 16×8 tile, packing the registers by hand from the table, calling `mma.sync`, writing back by the table, and comparing with the CPU. Its point is to let you verify the layout yourself:

```cuda title="mma_sync.cu"
// mma_sync.cu - PTX mma.sync.m16n8k16 directly, packing the registers by hand from the layout
// build: nvcc -O3 -arch=sm_80 mma_sync.cu -o mma_sync
#include "common.cuh"
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ uint32_t pack_half2(half lo, half hi) {
  __half2 h = __halves2half2(lo, hi);   // lo is the low 16 bits and holds the lower-numbered element
  return *reinterpret_cast<uint32_t*>(&h);
}

// A: 16x16 row-major, B: 16x8 row-major, D: 16x8 row-major (FP32)
__global__ void mma_16x8x16(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ D) {
  const int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  auto a = [&](int r, int c) { return A[r * 16 + c]; };
  auto b = [&](int r, int c) { return B[r * 8 + c]; };
  uint32_t ra[4], rb[2];
  ra[0] = pack_half2(a(g, 2 * t), a(g, 2 * t + 1));
  ra[1] = pack_half2(a(g + 8, 2 * t), a(g + 8, 2 * t + 1));
  ra[2] = pack_half2(a(g, 2 * t + 8), a(g, 2 * t + 9));
  ra[3] = pack_half2(a(g + 8, 2 * t + 8), a(g + 8, 2 * t + 9));
  rb[0] = pack_half2(b(2 * t, g), b(2 * t + 1, g));
  rb[1] = pack_half2(b(2 * t + 8, g), b(2 * t + 9, g));
  float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  D[g * 8 + 2 * t] = d[0];
  D[g * 8 + 2 * t + 1] = d[1];
  D[(g + 8) * 8 + 2 * t] = d[2];
  D[(g + 8) * 8 + 2 * t + 1] = d[3];
}

int main() {
  require_sm(8, 0);
  std::vector<float> fa(16 * 16), fb(16 * 8), ref(16 * 8, 0.f), got(16 * 8);
  fill_random(fa, 1);
  fill_random(fb, 2);
  std::vector<half> ha(fa.size()), hb(fb.size());
  for (size_t i = 0; i < fa.size(); ++i) { ha[i] = __float2half(fa[i]); fa[i] = __half2float(ha[i]); }
  for (size_t i = 0; i < fb.size(); ++i) { hb[i] = __float2half(fb[i]); fb[i] = __half2float(hb[i]); }
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 16; ++k)
      for (int j = 0; j < 8; ++j) ref[i * 8 + j] += fa[i * 16 + k] * fb[k * 8 + j];

  half *dA, *dB;
  float* dD;
  CUDA_CHECK(cudaMalloc(&dA, ha.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB, hb.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dD, got.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, ha.data(), ha.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hb.data(), hb.size() * sizeof(half), cudaMemcpyHostToDevice));
  mma_16x8x16<<<1, 32>>>(dA, dB, dD);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dD, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(got.data(), ref.data(), got.size(), 1e-3f, 1e-3f);
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dD));
  return ok ? 0 : 1;
}
```

### ldmatrix: loading fragments efficiently from shared memory {#ldmatrix从共享内存高效装载-fragment}

The program above loads registers one half at a time from global memory purely to demonstrate. A real kernel puts the data in shared memory first and loads it with **`ldmatrix`**: one `ldmatrix.sync.aligned.m8n8.x4.shared.b16` has a warp read four 8×8 half matrices from shared memory, and the elements each thread receives **match mma.sync's A operand layout exactly**. Each thread only supplies one row's start address: lanes 0-7 give the 8 rows of the first matrix, lanes 8-15 the second, and so on.

```cuda
uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[row][col]));
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
             : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3]) : "r"(smem_addr));
```

`ldmatrix` reads 16 bytes per row, and if the 8 row addresses map onto the same bank group they conflict. That is why high-performance kernels **swizzle** shared memory (scrambling the address with an xor). The `.trans` modifier transposes during the load, which is how the B operand is loaded.

## Hopper: wgmma {#hopperwgmma}

Hopper introduced the **warpgroup-level** asynchronous matrix instruction `wgmma.mma_async`: 4 consecutive warps (128 threads) together execute an m64nNk16 matrix multiply (N up to 256), with A from either registers or shared memory and B necessarily from shared memory (named by a 64-bit "matrix descriptor" giving the address and the swizzle mode). It is **asynchronous**: after issuing it a thread can do something else and wait later with `wgmma.commit_group` and `wgmma.wait_group`.

Combined with TMA (moving tiles into shared memory asynchronously) and warp specialization (some warps only move data and others only compute), GEMM and attention on Hopper reach above 90% of peak. Writing wgmma by hand is tedious, and in practice it is used through CUTLASS 3.x / CuTe or libraries like ThunderKittens. FlashAttention-3 is built on these features; see the next chapter, [asynchronous copies and Hopper/Blackwell](async-hopper.md).

!!! interview "How to explain it"
    On Tensor Cores: one matrix instruction is issued by a warp (a warpgroup on Hopper) and completes a small matrix multiply-accumulate (`mma.sync.m16n8k16`), with an order of magnitude more throughput than CUDA Cores (312 TFLOPS of BF16 against 19.5 of FP32 on an A100); inputs are low precision and the accumulator FP32. WMMA is simple but opaque, `mma.sync` has an explicit layout used with `ldmatrix` and a swizzle; the hierarchy is global memory, shared memory, register fragments, the matrix instruction. On Hopper it is wgmma plus TMA plus warp specialization, used in practice through CUTLASS / CuTe.

## Exercises {#练习}

**1. A calculation.** With WMMA's 16×16×16 fragments, how many floating-point operations does one `mma_sync` do? To sustain 312 TFLOPS on an A100, how many such warp-level instructions must the whole GPU execute per second?

??? success "Answer"
    16 × 16 × 16 = 4096 multiply-adds = **8192 floating-point operations**. 312e12 / 8192 ≈ 3.8e10 per second, which across 108 SMs at about 1.41 GHz is about 0.25 per SM per cycle, that is, one 16×16×16 matrix multiply per SM every 4 cycles. Sustaining that requires the data supply (shared memory to registers) to keep up, which is why Tensor Core kernels demand so much of their data movement.

**2. Reading the layout.** From the mma.sync m16n8k16 table, which 4 elements of C does lane 13 hold?

??? success "Answer"
    Lane 13: g = 13 / 4 = 3, t = 13 % 4 = 1. It holds c0 = (3, 2), c1 = (3, 3), c2 = (11, 2), c3 = (11, 3). The two elements of one row sit consecutively in one thread, which lets the write-back use a `float2` for 8 bytes at a time.

**3. Improving wmma_v2.** List at least three ways to make `wmma_v2` faster.

??? success "Answer"
    - **double buffering / a multi-stage cp.async pipeline**: load the next tile asynchronously while computing the current one; see the next chapter;
    - **larger warp tiles**: 64×64 per warp (4×4 fragments) to raise the fragment reuse, with 128×128 or 128×256 per block;
    - **mma.sync plus ldmatrix instead of WMMA**, with a shared-memory swizzle to remove bank conflicts;
    - **a better write-back**: stage the accumulators through shared memory and write them out vectorized, or convert to BF16 on output to move fewer bytes;
    - on Hopper, switch to TMA plus wgmma plus warp specialization.

## Summary {#小结}

- [x] A Tensor Core executes a small matrix multiply-accumulate per warp (per warpgroup on Hopper), with an order of magnitude more throughput than CUDA Cores.
- [x] Low-precision inputs with FP32 accumulation; BF16 is the mainstream LLM format, and FP8/FP4 need scaling factors.
- [x] WMMA is simple but opaque; mma.sync has an explicit layout used with ldmatrix and a swizzle.
- [x] A Tensor Core GEMM's hierarchy: global memory, shared memory, register fragments, the matrix instruction.
- [x] Hopper uses wgmma plus TMA plus warp specialization, reached in practice through CUTLASS/CuTe.
