// histogram.cu —— 256 个桶的直方图：全局原子 vs 共享内存私有直方图
// 编译：nvcc -O3 -arch=sm_75 histogram.cu -o histogram
#include "common.cuh"
#include <cstdint>

constexpr int kBins = 256;

__global__ void hist_global(const uint8_t* __restrict__ data, int n, unsigned* __restrict__ hist) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    atomicAdd(&hist[data[i]], 1u);
}

__global__ void hist_shared(const uint8_t* __restrict__ data, int n, unsigned* __restrict__ hist) {
  __shared__ unsigned local[kBins];
  for (int b = threadIdx.x; b < kBins; b += blockDim.x) local[b] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    atomicAdd(&local[data[i]], 1u);           // 共享内存原子操作，只在本 block 内争用
  __syncthreads();
  for (int b = threadIdx.x; b < kBins; b += blockDim.x)
    if (local[b]) atomicAdd(&hist[b], local[b]);   // 每个 block 每个桶最多一次全局原子操作
}

int main() {
  const int n = 1 << 26;
  std::vector<uint8_t> h(n);
  std::mt19937 gen(3);
  std::normal_distribution<float> dist(128.f, 20.f);   // 集中在中间的桶，争用更激烈
  for (auto& v : h) v = static_cast<uint8_t>(std::clamp(dist(gen), 0.f, 255.f));
  std::vector<unsigned> ref(kBins, 0), got(kBins);
  for (auto v : h) ref[v]++;

  uint8_t* d_data;
  unsigned* d_hist;
  CUDA_CHECK(cudaMalloc(&d_data, n));
  CUDA_CHECK(cudaMalloc(&d_hist, kBins * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(d_data, h.data(), n, cudaMemcpyHostToDevice));
  const int threads = 256, blocks = sm_count() * 8;

  bool ok = true;
  auto check = [&](const char* name) {
    CUDA_CHECK(cudaMemcpy(got.data(), d_hist, kBins * sizeof(unsigned), cudaMemcpyDeviceToHost));
    bool same = got == ref;
    std::printf("%s: %s\n", name, same ? "PASS" : "FAIL");
    ok &= same;
  };
  CUDA_CHECK(cudaMemset(d_hist, 0, kBins * sizeof(unsigned)));
  hist_global<<<blocks, threads>>>(d_data, n, d_hist);
  CUDA_CHECK_LAST();
  check("hist_global");
  CUDA_CHECK(cudaMemset(d_hist, 0, kBins * sizeof(unsigned)));
  hist_shared<<<blocks, threads>>>(d_data, n, d_hist);
  CUDA_CHECK_LAST();
  check("hist_shared");

  float t1 = time_ms([&] { hist_global<<<blocks, threads>>>(d_data, n, d_hist); });
  float t2 = time_ms([&] { hist_shared<<<blocks, threads>>>(d_data, n, d_hist); });
  std::printf("global atomics: %.3f ms (%.1f GB/s)\nshared atomics: %.3f ms (%.1f GB/s)\n",
              t1, gbps(n, t1), t2, gbps(n, t2));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_hist));
  return ok ? 0 : 1;
}
