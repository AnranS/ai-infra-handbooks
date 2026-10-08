// rmsnorm.cu —— RMSNorm 与融合了残差加法的 fused_add_rms_norm
// 编译：nvcc -O3 -arch=sm_75 rmsnorm.cu -o rmsnorm
#include "common.cuh"

__device__ __forceinline__ float warp_sum(float v) {
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  return v;
}
__device__ float block_sum(float v) {
  __shared__ float buf[32];
  __shared__ float total;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  v = warp_sum(v);
  if (lane == 0) buf[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = warp_sum(lane < blockDim.x / 32 ? buf[lane] : 0.f);
    if (lane == 0) total = v;
  }
  __syncthreads();
  return total;
}

// 一个 block 一行；每个线程把自己负责的元素缓存在寄存器里（最多 kMaxPerThread 个）
constexpr int kThreads = 256;
constexpr int kMaxPerThread = 32;   // 支持 hidden <= 8192

__global__ void rms_norm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                         int hidden, float eps) {
  const size_t off = static_cast<size_t>(blockIdx.x) * hidden;
  float v[kMaxPerThread];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    v[k] = c < hidden ? x[off + c] : 0.f;
    ss += v[k] * v[k];
  }
  const float scale = rsqrtf(block_sum(ss) / hidden + eps);
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    if (c < hidden) y[off + c] = v[k] * scale * w[c];
  }
}

// residual += hidden；hidden = rms_norm(residual) * w。两个张量都原地更新
__global__ void fused_add_rms_norm(float* __restrict__ hidden, float* __restrict__ residual,
                                   const float* __restrict__ w, int n_hidden, float eps) {
  const size_t off = static_cast<size_t>(blockIdx.x) * n_hidden;
  float v[kMaxPerThread];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    if (c < n_hidden) {
      v[k] = residual[off + c] + hidden[off + c];
      residual[off + c] = v[k];            // 新的残差写回，供下一层使用
    } else {
      v[k] = 0.f;
    }
    ss += v[k] * v[k];
  }
  const float scale = rsqrtf(block_sum(ss) / n_hidden + eps);
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    if (c < n_hidden) hidden[off + c] = v[k] * scale * w[c];
  }
}

int main() {
  const int tokens = 512, hidden = 4096;
  const float eps = 1e-6f;
  const size_t n = static_cast<size_t>(tokens) * hidden, bytes = n * sizeof(float);
  std::vector<float> hx(n), hr(n), hw(hidden), ref_y(n), ref_r(n), got(n);
  fill_random(hx, 1);
  fill_random(hr, 2);
  fill_random(hw, 3, 0.5f, 1.5f);

  // CPU 参考：先算 RMSNorm(x)，再算融合版本
  for (int t = 0; t < tokens; ++t) {
    double ss = 0;
    for (int c = 0; c < hidden; ++c) ss += double(hx[t * hidden + c]) * hx[t * hidden + c];
    float scale = 1.f / std::sqrt(static_cast<float>(ss / hidden) + eps);
    for (int c = 0; c < hidden; ++c) ref_y[t * hidden + c] = hx[t * hidden + c] * scale * hw[c];
  }

  float *dx, *dr, *dw, *dy;
  CUDA_CHECK(cudaMalloc(&dx, bytes));
  CUDA_CHECK(cudaMalloc(&dr, bytes));
  CUDA_CHECK(cudaMalloc(&dy, bytes));
  CUDA_CHECK(cudaMalloc(&dw, hidden * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dr, hr.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, hw.data(), hidden * sizeof(float), cudaMemcpyHostToDevice));

  rms_norm<<<tokens, kThreads>>>(dx, dw, dy, hidden, eps);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dy, bytes, cudaMemcpyDeviceToHost));
  std::printf("rms_norm            ");
  bool ok = check_close(got.data(), ref_y.data(), n, 1e-4f, 1e-5f);

  // 融合版本的参考结果
  for (int t = 0; t < tokens; ++t) {
    double ss = 0;
    for (int c = 0; c < hidden; ++c) {
      float r = hr[t * hidden + c] + hx[t * hidden + c];
      ref_r[t * hidden + c] = r;
      ss += double(r) * r;
    }
    float scale = 1.f / std::sqrt(static_cast<float>(ss / hidden) + eps);
    for (int c = 0; c < hidden; ++c) ref_y[t * hidden + c] = ref_r[t * hidden + c] * scale * hw[c];
  }
  fused_add_rms_norm<<<tokens, kThreads>>>(dx, dr, dw, hidden, eps);   // 原地修改 dx 和 dr
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dx, bytes, cudaMemcpyDeviceToHost));
  std::printf("fused (hidden out)  ");
  ok &= check_close(got.data(), ref_y.data(), n, 1e-4f, 1e-5f);
  CUDA_CHECK(cudaMemcpy(got.data(), dr, bytes, cudaMemcpyDeviceToHost));
  std::printf("fused (residual)    ");
  ok &= check_close(got.data(), ref_r.data(), n, 1e-6f, 1e-6f);

  float t1 = time_ms([&] { rms_norm<<<tokens, kThreads>>>(dx, dw, dy, hidden, eps); });
  float t2 = time_ms([&] { fused_add_rms_norm<<<tokens, kThreads>>>(dx, dr, dw, hidden, eps); });
  std::printf("rms_norm: %.3f ms (%.1f GB/s)\nfused_add_rms_norm: %.3f ms (%.1f GB/s)\n",
              t1, gbps(2.0 * bytes, t1), t2, gbps(4.0 * bytes, t2));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dr));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dw));
  return ok ? 0 : 1;
}
