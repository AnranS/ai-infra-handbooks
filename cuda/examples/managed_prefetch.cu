// managed_prefetch.cu —— 统一内存 + 预取；演示如何同时兼容 CUDA 12 与 CUDA 13 的 API
// 编译：nvcc -O3 -arch=sm_75 managed_prefetch.cu -o managed_prefetch
#include "common.cuh"

__global__ void square(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * x[i];
}

void prefetch(const void* p, size_t bytes, int device, cudaStream_t s) {
#if CUDART_VERSION >= 13000
  cudaMemLocation loc{};
  loc.type = device == cudaCpuDeviceId ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
  loc.id = device == cudaCpuDeviceId ? 0 : device;
  CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, loc, 0, s));
#else
  CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, device, s));
#endif
}

int main() {
  const int n = 1 << 24;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float* x;
  CUDA_CHECK(cudaMallocManaged(&x, bytes));
  for (int i = 0; i < n; ++i) x[i] = static_cast<float>(i % 1000) * 0.01f;   // 在 CPU 上直接写

  int dev = 0, concurrent = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev));
  if (concurrent) prefetch(x, bytes, dev, 0);   // 一次性迁移到 GPU，避免 kernel 里大量缺页

  square<<<(n + 255) / 256, 256>>>(x, n);
  CUDA_CHECK_LAST();
  if (concurrent) prefetch(x, bytes, cudaCpuDeviceId, 0);   // 迁移回 CPU
  CUDA_CHECK(cudaDeviceSynchronize());                       // 访问前必须同步

  size_t bad = 0;
  for (int i = 0; i < n; ++i) {
    float v = static_cast<float>(i % 1000) * 0.01f;
    bad += std::fabs(x[i] - v * v) > 1e-4f;
  }
  std::printf("%s (%zu mismatches)\n", bad ? "FAIL" : "PASS", bad);
  CUDA_CHECK(cudaFree(x));
  return bad ? 1 : 0;
}
