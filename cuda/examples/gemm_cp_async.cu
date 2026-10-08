// gemm_cp_async.cu —— 在二维寄存器分块 SGEMM 上加入 3 级 cp.async 流水
// 编译：nvcc -O3 -arch=sm_80 gemm_cp_async.cu -o gemm_cp_async
// 要求：M、N 是 128 的倍数，K 是 8 的倍数
#include "common.cuh"
#include <cuda_pipeline.h>

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8, STAGES = 3;
constexpr int kThreads = (BM * BN) / (TM * TN);   // 256

__global__ void __launch_bounds__(kThreads)
sgemm_cp_async(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
               float beta, float* __restrict__ C) {
  __shared__ __align__(16) float As[STAGES][BM * BK];
  __shared__ __align__(16) float Bs[STAGES][BK * BN];
  const int tid = threadIdx.x;
  const int threadCol = tid % (BN / TN), threadRow = tid / (BN / TN);
  // 每个线程每块负责搬 A 的一个 16 字节和 B 的一个 16 字节
  const int a_row = tid / (BK / 4), a_col = (tid % (BK / 4)) * 4;   // A 块 128x8：每行 2 个 16 字节
  const int b_row = tid / (BN / 4), b_col = (tid % (BN / 4)) * 4;   // B 块 8x128：每行 32 个 16 字节
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  auto load_tile = [&](int buf, int kt) {
    const int k0 = kt * BK;
    __pipeline_memcpy_async(&As[buf][a_row * BK + a_col], &A[a_row * K + k0 + a_col], 16);
    __pipeline_memcpy_async(&Bs[buf][b_row * BN + b_col], &B[(k0 + b_row) * N + b_col], 16);
  };

  const int num_tiles = K / BK;
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < num_tiles) load_tile(s, s);
    __pipeline_commit();
  }

  float acc[TM][TN] = {{0.f}};
  float regM[TM], regN[TN];
  for (int kt = 0; kt < num_tiles; ++kt) {
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < num_tiles) load_tile(next % STAGES, next);
    __pipeline_commit();

    const float* as = As[kt % STAGES];
    const float* bs = Bs[kt % STAGES];
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = as[(threadRow * TM + i) * BK + dot];
#pragma unroll
      for (int i = 0; i < TN; ++i) regN[i] = bs[dot * BN + threadCol * TN + i];
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) acc[m][n] += regM[m] * regN[n];
    }
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      float& c = C[(threadRow * TM + m) * N + threadCol * TN + n];
      c = alpha * acc[m][n] + beta * c;
    }
}

int main(int argc, char** argv) {
  require_sm(8, 0);
  const int check_n = 256;
  const int n_bench = argc > 1 ? std::atoi(argv[1]) : 4096;
  const float alpha = 1.f, beta = 0.f;
  {
    const int n = check_n;
    std::vector<float> hA(n * n), hB(n * n), ref(n * n, 0.f), got(n * n);
    fill_random(hA, 1);
    fill_random(hB, 2);
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j) ref[i * n + j] += hA[i * n + k] * hB[k * n + j];
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(C, 0, n * n * sizeof(float)));
    sgemm_cp_async<<<dim3(n / BN, n / BM), kThreads>>>(n, n, n, alpha, A, B, beta, C);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(got.data(), C, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = check_close(got.data(), ref.data(), n * n, 1e-4f, 1e-4f);
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    if (!ok) return 1;
  }
  const int n = n_bench;
  const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, bytes));
  CUDA_CHECK(cudaMalloc(&B, bytes));
  CUDA_CHECK(cudaMalloc(&C, bytes));
  CUDA_CHECK(cudaMemset(A, 0, bytes));
  CUDA_CHECK(cudaMemset(B, 0, bytes));
  float ms = time_ms([&] { sgemm_cp_async<<<dim3(n / BN, n / BM), kThreads>>>(n, n, n, alpha, A, B, beta, C); });
  std::printf("cp.async 3-stage SGEMM, n=%d: %.3f ms, %.2f TFLOPS (compare with gemm v4)\n", n, ms,
              tflops(2.0 * n * n * n, ms));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
