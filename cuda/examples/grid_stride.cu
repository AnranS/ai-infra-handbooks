// grid_stride.cu —— 网格跨步循环实现 SAXPY: y = a * x + y
// 编译：nvcc -O3 -arch=sm_75 grid_stride.cu -o grid_stride
#include "common.cuh"

__global__ void saxpy(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < static_cast<size_t>(n);
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  const int n = 1 << 25;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> h_x(n), h_y(n), ref(n);
  fill_random(h_x, 1);
  fill_random(h_y, 2);
  const float a = 2.5f;
  for (int i = 0; i < n; ++i) ref[i] = a * h_x[i] + h_y[i];

  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

  // block 数只和 GPU 规模有关，和数据量无关：每个 SM 放若干个 block
  const int threads = 256;
  const int blocks = sm_count() * 8;
  saxpy<<<blocks, threads>>>(n, a, d_x, d_y);
  CUDA_CHECK_LAST();
  std::vector<float> h_out(n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  check_close(h_out.data(), ref.data(), n);

  // 计时会反复更新 y，只关心速度
  float ms = time_ms([&] { saxpy<<<blocks, threads>>>(n, a, d_x, d_y); });
  std::printf("saxpy (grid-stride, %d blocks): %.3f ms, %.1f GB/s\n", blocks, ms, gbps(3.0 * bytes, ms));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return 0;
}
