// GEMM 题的性能参照：同形状的 cuBLAS SGEMM（FP32，行主序）。只在真 GPU 上编译，判题时自动链接 -lcublas。
#pragma once
#include "judge.cuh"

#ifndef PRACTICE_EMU
#include <cublas_v2.h>
#endif

namespace pj {

// 返回 cuBLAS 算 C = A·B（A: M×K，B: K×N，都是行主序）的平均耗时（毫秒）；模拟器上返回 0。
inline float cublas_sgemm_ms(const float* A, const float* B, float* C, int M, int N, int K) {
#ifdef PRACTICE_EMU
  (void)A, (void)B, (void)C, (void)M, (void)N, (void)K;
  return 0.f;
#else
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 0.f;
  const float one = 1.f, zero = 0.f;
  // 行主序的 C = A·B 等价于列主序的 Cᵀ = Bᵀ·Aᵀ
  auto run = [&] { cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N); };
  float ms = timeit(run, 10);
  cublasDestroy(h);
  return ms;
#endif
}

// GEMM 按 cuBLAS 定档（铜 40%、银 70%、金 90%）
inline void cublas_tier(const char* name, float ms, const float* A, const float* B, float* C, int M, int N, int K) {
  if (kEmulator || ms <= 0) return;
  float ref = cublas_sgemm_ms(A, B, C, M, N, K);
  if (ref > 0) tier(name, ref / ms, 0.4, 0.7, 0.9, "同形状 cuBLAS SGEMM");
}

}  // namespace pj
