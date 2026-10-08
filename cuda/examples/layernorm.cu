// layernorm.cu —— 一个 block 一行，寄存器缓存整行，两遍法求均值和方差
// 编译：nvcc -O3 -arch=sm_75 layernorm.cu -o layernorm
#include "common.cuh"

constexpr int kThreads = 256, kMaxPerThread = 32;

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
  float r = total;
  __syncthreads();   // 同一个 kernel 里会调用两次，读完再允许下一次覆盖
  return r;
}

__global__ void layer_norm(const float* __restrict__ x, const float* __restrict__ gamma,
                           const float* __restrict__ beta, float* __restrict__ y, int H, float eps) {
  const size_t off = static_cast<size_t>(blockIdx.x) * H;
  float v[kMaxPerThread];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    v[k] = c < H ? x[off + c] : 0.f;
    s += v[k];
  }
  const float mean = block_sum(s) / H;
  float sq = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    float d = c < H ? v[k] - mean : 0.f;
    sq += d * d;
  }
  const float rstd = rsqrtf(block_sum(sq) / H + eps);
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    int c = k * kThreads + threadIdx.x;
    if (c < H) y[off + c] = (v[k] - mean) * rstd * gamma[c] + beta[c];
  }
}

int main() {
  const int rows = 256, H = 5120;
  const float eps = 1e-5f;
  const size_t n = static_cast<size_t>(rows) * H;
  std::vector<float> hx(n), hg(H), hb(H), ref(n), got(n);
  fill_random(hx, 1, 100.f, 102.f);   // 均值远大于标准差，检验数值稳定性
  fill_random(hg, 2, 0.5f, 1.5f);
  fill_random(hb, 3);
  for (int r = 0; r < rows; ++r) {
    double mean = 0, var = 0;
    for (int c = 0; c < H; ++c) mean += hx[r * H + c];
    mean /= H;
    for (int c = 0; c < H; ++c) var += (hx[r * H + c] - mean) * (hx[r * H + c] - mean);
    var /= H;
    for (int c = 0; c < H; ++c)
      ref[r * H + c] = static_cast<float>((hx[r * H + c] - mean) / std::sqrt(var + eps) * hg[c] + hb[c]);
  }
  float *dx, *dg, *db, *dy;
  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dg, H * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, H * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dg, hg.data(), H * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), H * sizeof(float), cudaMemcpyHostToDevice));
  layer_norm<<<rows, kThreads>>>(dx, dg, db, dy, H, eps);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(got.data(), ref.data(), n, 1e-3f, 1e-3f);
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(db));
  return ok ? 0 : 1;
}
