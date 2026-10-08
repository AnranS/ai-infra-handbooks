// nvtx_demo.cu —— 用 NVTX 标注程序阶段，在 Nsight Systems 时间线上查看
// 编译：nvcc -O3 -arch=sm_75 nvtx_demo.cu -o nvtx_demo
// 分析：nsys profile --trace=cuda,nvtx -o nvtx_demo ./nvtx_demo
#include "common.cuh"
#include <nvtx3/nvToolsExt.h>

__global__ void scale(float* x, int n, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= s;
}

int main() {
  const int n = 1 << 24;
  std::vector<float> h(n, 1.f);
  float* d;

  nvtxRangePushA("setup");
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  nvtxRangePop();

  nvtxRangePushA("compute");
  for (int step = 0; step < 10; ++step) {
    nvtxRangePushA("step");
    scale<<<(n + 255) / 256, 256>>>(d, n, 1.01f);
    CUDA_CHECK_LAST();
    nvtxRangePop();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  const float expect = std::pow(1.01f, 10.f);
  const bool ok = std::fabs(h[0] - expect) < 1e-4f && std::fabs(h[n - 1] - expect) < 1e-4f;
  std::printf("%s (x = %.6f, expected %.6f)\n", ok ? "PASS" : "FAIL", h[0], expect);
  CUDA_CHECK(cudaFree(d));
  return ok ? 0 : 1;
}
