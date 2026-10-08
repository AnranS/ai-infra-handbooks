// divergence.cu —— warp 内分支发散的代价
// 编译：nvcc -O3 -arch=sm_75 divergence.cu -o divergence
#include "common.cuh"

__device__ __forceinline__ float heavy(float x, int iters, float c) {
  for (int k = 0; k < iters; ++k) x = x * c + 0.5f;
  return x;
}

// 同一个 warp 内奇偶线程走不同分支
__global__ void divergent(float* out, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = static_cast<float>(threadIdx.x);
  if (threadIdx.x % 2 == 0) x = heavy(x, iters, 0.999f);
  else                      x = heavy(x, iters, 0.998f);
  out[i] = x;
}

// 分支条件以 warp 为单位，同一个 warp 走同一条路
__global__ void uniform(float* out, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = static_cast<float>(threadIdx.x);
  if ((threadIdx.x / 32) % 2 == 0) x = heavy(x, iters, 0.999f);
  else                             x = heavy(x, iters, 0.998f);
  out[i] = x;
}

int main() {
  const int threads = 256, blocks = sm_count() * 8, iters = 2000;
  const int n = threads * blocks;
  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

  // 正确性：两个 kernel 每个线程做的计算完全相同，只是分组方式不同
  std::vector<float> h(n), ref(n);
  uniform<<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int t = 0; t < n; ++t) {
    int tid = t % threads;
    float x = static_cast<float>(tid), c = ((tid / 32) % 2 == 0) ? 0.999f : 0.998f;
    for (int k = 0; k < iters; ++k) x = x * c + 0.5f;
    ref[t] = x;
  }
  bool ok = check_close(h.data(), ref.data(), n, 1e-3f, 1e-3f);

  float t_div = time_ms([&] { divergent<<<blocks, threads>>>(d_out, iters); });
  float t_uni = time_ms([&] { uniform<<<blocks, threads>>>(d_out, iters); });
  std::printf("divergent: %.3f ms\nuniform  : %.3f ms\nratio    : %.2fx\n", t_div, t_uni, t_div / t_uni);
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
