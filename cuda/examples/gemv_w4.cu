// gemv_w4.cu —— FP32 GEMV 与 W4A16 风格的 INT4 按组量化 GEMV（一个 warp 负责一行）
// 编译：nvcc -O3 -arch=sm_75 gemv_w4.cu -o gemv_w4
// 为了让示例聚焦在解包与反量化上，激活和缩放因子都用 FP32；真实 kernel 用 FP16/BF16
#include "common.cuh"
#include <cstdint>

constexpr int GROUP = 128;       // 每 128 个 K 方向的元素共享一个缩放因子和零点
constexpr int kWarps = 4;        // 每个 block 4 个 warp，每个 warp 一行

__device__ __forceinline__ float warp_sum(float v) {
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  return v;
}

// 基线：FP32 权重，每个 lane 每次读一个 float4
__global__ void gemv_fp32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N, int K) {
  const int row = blockIdx.x * kWarps + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= N) return;
  const float4* w4 = reinterpret_cast<const float4*>(W + static_cast<size_t>(row) * K);
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float acc = 0.f;
  for (int i = lane; i < K / 4; i += 32) {
    const float4 w = w4[i], v = x4[i];
    acc += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

// INT4：W 打包成 [N, K/8] 个 uint32；scales、zeros 为 [N, K/GROUP]
__global__ void gemv_w4(const uint32_t* __restrict__ Wq, const float* __restrict__ scales,
                        const float* __restrict__ zeros, const float* __restrict__ x, float* __restrict__ y,
                        int N, int K) {
  extern __shared__ float xs[];          // x 被整个 block 共用，先放进共享内存
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();

  const int row = blockIdx.x * kWarps + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= N) return;                  // 同步之后再退出，不影响 __syncthreads
  const uint32_t* wrow = Wq + static_cast<size_t>(row) * (K / 8);
  const float* srow = scales + static_cast<size_t>(row) * (K / GROUP);
  const float* zrow = zeros + static_cast<size_t>(row) * (K / GROUP);
  float acc = 0.f;
  for (int i = lane; i < K / 8; i += 32) {            // 相邻 lane 读相邻的 32 位字：合并访问
    const uint32_t word = wrow[i];
    const int k = i * 8, g = k / GROUP;
    const float s = srow[g], z = zrow[g];
    const float* xv = xs + k;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float q = static_cast<float>((word >> (4 * j)) & 0xFu);
      acc += (q - z) * s * xv[j];
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = acc;
}

int main() {
  const int N = 4096, K = 4096;   // K 需要是 GROUP 的整数倍
  std::vector<float> W(static_cast<size_t>(N) * K), x(K);
  fill_random(W, 1);
  fill_random(x, 2);

  // 离线量化：每组 128 个元素，非对称 4 位
  const int groups = K / GROUP;
  std::vector<uint32_t> Wq(static_cast<size_t>(N) * K / 8, 0u);
  std::vector<float> scales(static_cast<size_t>(N) * groups), zeros(scales.size()), W_deq(W.size());
  for (int r = 0; r < N; ++r)
    for (int g = 0; g < groups; ++g) {
      const float* w = &W[static_cast<size_t>(r) * K + g * GROUP];
      float lo = *std::min_element(w, w + GROUP), hi = *std::max_element(w, w + GROUP);
      float s = (hi - lo) / 15.f;
      if (s == 0.f) s = 1.f;
      float z = std::round(-lo / s);
      scales[static_cast<size_t>(r) * groups + g] = s;
      zeros[static_cast<size_t>(r) * groups + g] = z;
      for (int j = 0; j < GROUP; ++j) {
        const int k = g * GROUP + j;
        int q = static_cast<int>(std::round(w[j] / s + z));
        q = std::min(15, std::max(0, q));
        Wq[(static_cast<size_t>(r) * K + k) / 8] |= static_cast<uint32_t>(q) << (4 * (k % 8));
        W_deq[static_cast<size_t>(r) * K + k] = (q - z) * s;
      }
    }

  // 参考结果：FP32 GEMV，以及"用反量化后的权重"做的 GEMV（INT4 kernel 应与后者一致）
  std::vector<float> ref_fp32(N), ref_q(N), got(N);
  double qerr = 0;
  for (int r = 0; r < N; ++r) {
    double a = 0, b = 0;
    for (int k = 0; k < K; ++k) {
      a += double(W[static_cast<size_t>(r) * K + k]) * x[k];
      b += double(W_deq[static_cast<size_t>(r) * K + k]) * x[k];
    }
    ref_fp32[r] = static_cast<float>(a);
    ref_q[r] = static_cast<float>(b);
    qerr = std::max(qerr, std::fabs(a - b));
  }
  std::printf("max |y_fp32 - y_int4| on CPU (quantization error): %.4f\n", qerr);

  float *dW, *dx, *dy, *ds, *dz;
  uint32_t* dWq;
  CUDA_CHECK(cudaMalloc(&dW, W.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dWq, Wq.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&ds, scales.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dz, zeros.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dx, K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWq, Wq.data(), Wq.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ds, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dz, zeros.data(), zeros.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), K * sizeof(float), cudaMemcpyHostToDevice));

  const int blocks = (N + kWarps - 1) / kWarps;
  bool ok = true;
  gemv_fp32<<<blocks, kWarps * 32>>>(dW, dx, dy, N, K);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("gemv_fp32 ");
  ok &= check_close(got.data(), ref_fp32.data(), N, 1e-3f, 1e-3f);
  gemv_w4<<<blocks, kWarps * 32, K * sizeof(float)>>>(dWq, ds, dz, dx, dy, N, K);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("gemv_w4   ");
  ok &= check_close(got.data(), ref_q.data(), N, 1e-3f, 1e-3f);

  const double fp32_bytes = double(W.size()) * 4;
  const double w4_bytes = double(Wq.size()) * 4 + double(scales.size()) * 8;
  float t1 = time_ms([&] { gemv_fp32<<<blocks, kWarps * 32>>>(dW, dx, dy, N, K); });
  float t2 = time_ms([&] { gemv_w4<<<blocks, kWarps * 32, K * sizeof(float)>>>(dWq, ds, dz, dx, dy, N, K); });
  std::printf("fp32 weights: %.3f ms (%.1f GB/s)\nint4 weights: %.3f ms (%.1f GB/s), speedup %.2fx\n",
              t1, gbps(fp32_bytes, t1), t2, gbps(w4_bytes, t2), t1 / t2);
  CUDA_CHECK(cudaFree(dW));
  CUDA_CHECK(cudaFree(dWq));
  CUDA_CHECK(cudaFree(ds));
  CUDA_CHECK(cudaFree(dz));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  return ok ? 0 : 1;
}
