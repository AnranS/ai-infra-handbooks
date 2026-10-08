// wmma_gemm.cu —— 用 WMMA 调用 Tensor Core：FP16 输入、FP32 累加
// 编译：nvcc -O3 -arch=sm_75 wmma_gemm.cu -o wmma_gemm
// 要求：M、N 是 64 的倍数，K 是 32 的倍数
#include "common.cuh"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// v1：一个 warp 算一个 16x16 的 C 块，fragment 直接从全局内存加载
__global__ void wmma_v1(int M, int N, int K, const half* __restrict__ A, const half* __restrict__ B,
                        float* __restrict__ C) {
  const int warp = threadIdx.x / 32;
  const int tile_m = blockIdx.y;                  // 第几个 16 行
  const int tile_n = blockIdx.x * 4 + warp;       // 每个 block 4 个 warp，横向排开
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

// v2：block 负责 64x64，4 个 warp 按 2x2 排列，每个 warp 负责 32x32；A、B 先进共享内存
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
    // 64x32 的 A 块 = 256 个 uint4（每个 8 个 half），128 个线程每人搬 2 个
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      int idx = tid + t * 128, r = idx / 4, c = (idx % 4) * 8;
      *reinterpret_cast<uint4*>(&As[r][c]) = *reinterpret_cast<const uint4*>(&A[(row0 + r) * K + k0 + c]);
    }
    // 32x64 的 B 块同样是 256 个 uint4
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

  // CPU 参考：只检查前 64 行，避免 CPU 计算太久
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
