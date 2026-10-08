// block_sum.cu —— warp shuffle + 共享内存的两级归约，每个 block 一次全局原子加
// 编译：nvcc -O3 -arch=sm_75 block_sum.cu -o block_sum
#include "common.cuh"

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

// 要求 blockDim.x 是 32 的倍数，且不超过 1024
__device__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  v = warp_reduce_sum(v);                    // 第一级：warp 内
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  const int num_warps = blockDim.x / 32;
  v = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
  if (warp == 0) v = warp_reduce_sum(v);     // 第二级：第 0 个 warp 汇总各 warp 的结果
  return v;                                  // 只有线程 0 的值是整个 block 的和
}

__global__ void sum_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) v += x[i];
  v = block_reduce_sum(v);
  if (threadIdx.x == 0) atomicAdd(out, v);   // 每个 block 只做一次原子操作
}

int main() {
  const int n = 50'000'000;
  std::vector<float> h(n);
  fill_random(h, 7, 0.f, 1.f);
  double ref = 0.0;
  for (float v : h) ref += v;

  float *d_x, *d_out;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256, blocks = sm_count() * 4;
  auto run = [&] {
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
    sum_kernel<<<blocks, threads>>>(d_x, d_out, n);
  };
  run();
  CUDA_CHECK_LAST();
  float got = 0.f;
  CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  float ref_f = static_cast<float>(ref);
  bool ok = check_close(&got, &ref_f, 1, 1e-4f, 0.f);   // float 累加有舍入误差，允许 1e-4 的相对误差

  float ms = time_ms(run);
  std::printf("sum of %d floats: %.3f ms, %.1f GB/s\n", n, ms, gbps(n * sizeof(float), ms));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
