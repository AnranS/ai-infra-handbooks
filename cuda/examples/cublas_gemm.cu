// cublas_gemm.cu —— 用 cublasGemmEx 计算行主序的 FP16 GEMM（FP32 累加、FP32 输出）
// 编译：nvcc -O3 -arch=sm_75 cublas_gemm.cu -o cublas_gemm -lcublas
#include "common.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CUBLAS_CHECK(call)                                                           \
  do {                                                                               \
    cublasStatus_t s_ = (call);                                                      \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                               \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s_), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                       \
    }                                                                                \
  } while (0)

int main() {
  const int M = 512, N = 384, K = 256;
  std::vector<float> fa(M * K), fb(K * N), ref(M * N, 0.f), got(M * N);
  fill_random(fa, 1);
  fill_random(fb, 2);
  std::vector<half> ha(fa.size()), hb(fb.size());
  for (size_t i = 0; i < fa.size(); ++i) { ha[i] = __float2half(fa[i]); fa[i] = __half2float(ha[i]); }
  for (size_t i = 0; i < fb.size(); ++i) { hb[i] = __float2half(fb[i]); fb[i] = __half2float(hb[i]); }
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      for (int j = 0; j < N; ++j) ref[i * N + j] += fa[i * K + k] * fb[k * N + j];

  half *dA, *dB;
  float* dC;
  CUDA_CHECK(cudaMalloc(&dA, ha.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB, hb.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dC, got.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, ha.data(), ha.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hb.data(), hb.size() * sizeof(half), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.f, beta = 0.f;
  // 行主序 C(MxN) = A(MxK) B(KxN)  <=>  列主序 C^T(NxM) = B^T(NxK) A^T(KxM)
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            dB, CUDA_R_16F, N,     // 第一个矩阵传 B，leading dimension 是 B 的行长 N
                            dA, CUDA_R_16F, K,     // 第二个矩阵传 A，leading dimension 是 K
                            &beta, dC, CUDA_R_32F, N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(got.data(), ref.data(), got.size(), 1e-3f, 1e-3f);
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return ok ? 0 : 1;
}
