// conv1d.cu —— 常量内存存权重 + 共享内存缓存带光环的输入块
// 编译：nvcc -O3 -arch=sm_75 conv1d.cu -o conv1d
#include "common.cuh"

constexpr int R = 8;
constexpr int BLOCK = 256;
__constant__ float c_w[2 * R + 1];

__global__ void conv1d(const float* __restrict__ x, float* __restrict__ y, int n) {
  __shared__ float tile[BLOCK + 2 * R];
  const int base = blockIdx.x * BLOCK;
  // 协作加载：BLOCK + 2R 个元素，每个线程可能加载多个
  for (int j = threadIdx.x; j < BLOCK + 2 * R; j += blockDim.x) {
    int g = base + j - R;
    tile[j] = (g >= 0 && g < n) ? x[g] : 0.f;
  }
  __syncthreads();

  int i = base + threadIdx.x;
  if (i < n) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k <= 2 * R; ++k) acc += c_w[k] * tile[threadIdx.x + k];
    y[i] = acc;
  }
}

int main() {
  const int n = (1 << 22) + 123;
  std::vector<float> hx(n), hw(2 * R + 1), hy(n), ref(n);
  fill_random(hx, 1);
  fill_random(hw, 2);
  for (int i = 0; i < n; ++i) {
    float acc = 0.f;
    for (int k = -R; k <= R; ++k)
      if (i + k >= 0 && i + k < n) acc += hw[k + R] * hx[i + k];
    ref[i] = acc;
  }

  float *dx, *dy;
  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(c_w, hw.data(), hw.size() * sizeof(float)));

  const int blocks = (n + BLOCK - 1) / BLOCK;
  conv1d<<<blocks, BLOCK>>>(dx, dy, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(hy.data(), ref.data(), n, 1e-4f, 1e-5f);

  float ms = time_ms([&] { conv1d<<<blocks, BLOCK>>>(dx, dy, n); });
  std::printf("conv1d: %.3f ms, %.1f GB/s\n", ms, gbps(2.0 * n * sizeof(float), ms));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  return ok ? 0 : 1;
}
