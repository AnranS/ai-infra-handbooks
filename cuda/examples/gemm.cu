// gemm.cu —— SGEMM 从朴素实现到二维寄存器分块 + 向量化
// 编译：nvcc -O3 -arch=sm_80 gemm.cu -o gemm
// 对比 cuBLAS：nvcc -O3 -arch=sm_80 -DWITH_CUBLAS gemm.cu -o gemm -lcublas
// 要求：M、N、K 都是 128 的倍数（为了代码简洁，没有处理边界）
#include "common.cuh"
#ifdef WITH_CUBLAS
#include <cublas_v2.h>
#endif

// ---------------- v0：threadIdx.x 对应行，访存不合并 ----------------
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

// ---------------- v1：threadIdx.x 对应列，读 B / 写 C 合并 ----------------
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

// ---------------- v2：32x32 共享内存分块 ----------------
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

// ---------------- v3：一维寄存器分块，每线程算一列上的 TM 个元素 ----------------
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

// ---------------- v4：二维寄存器分块，每线程算 TM x TN ----------------
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

// ---------------- v5：v4 + A 转置存入共享内存 + float4 访存 ----------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_v5(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
         float beta, float* __restrict__ C) {
  constexpr int kThreads = (BM * BN) / (TM * TN);
  static_assert(BM * BK / 4 == kThreads && BK * BN / 4 == kThreads, "one float4 per thread per load");
  __shared__ __align__(16) float As[BK * BM];   // 转置存放：As[k][m]；float4 访问要求 16 字节对齐
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

// ---------------- 启动封装 ----------------
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

// CPU 参考实现：i-k-j 循环顺序，对缓存友好
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
  const int check_n = 512;                               // 正确性检查用小矩阵（CPU 算得快）
  const int bench_n = argc > 1 ? std::atoi(argv[1]) : 4096;
  const float alpha = 1.f, beta = 0.5f;

  struct { const char* name; Launcher fn; } versions[] = {
      {"v0 naive (x->row)", launch_v0},    {"v1 coalesced (x->col)", launch_v1},
      {"v2 smem tiling", launch_v2},        {"v3 1D register tiling", launch_v3},
      {"v4 2D register tiling", launch_v4}, {"v5 vectorized", launch_v5},
  };

  // ---- 正确性 ----
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

  // ---- 性能 ----
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
  // cuBLAS 是列主序：行主序的 C = A B 等价于列主序的 C^T = B^T A^T
  float ms_cublas = time_ms([&] {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n, A, n, &beta, C, n);
  }, 10, 2);
  std::printf("%-24s %10.3f %10.2f\n", "cuBLAS", ms_cublas, tflops(flops, ms_cublas));
  cublasDestroy(handle);
#endif
  for (auto& v : versions) {
    const int iters = (v.fn == launch_v0) ? 1 : 10;   // v0 太慢，只跑一次
    float ms = time_ms([&] { v.fn(n, n, n, alpha, A, B, beta, C); }, iters, 1);
    std::printf("%-24s %10.3f %10.2f\n", v.name, ms, tflops(flops, ms));
  }
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
