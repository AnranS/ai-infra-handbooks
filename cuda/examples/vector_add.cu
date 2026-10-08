// vector_add.cu —— 第一个完整的 CUDA 程序
// 编译：nvcc -O3 -arch=sm_75 vector_add.cu -o vector_add
#include "common.cuh"

// 每个线程负责一个元素
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程编号
  if (i < n) {                                    // 最后一个 block 可能有多余的线程
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int n = 1 << 24;  // 约 1600 万个元素
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  // 1. 准备主机数据和 CPU 参考结果
  std::vector<float> h_a(n), h_b(n), h_c(n), ref(n);
  fill_random(h_a, 1);
  fill_random(h_b, 2);
  for (int i = 0; i < n; ++i) ref[i] = h_a[i] + h_b[i];

  // 2. 分配显存并拷贝输入
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  // 3. 启动 kernel：向上取整，保证覆盖所有元素
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
  CUDA_CHECK_LAST();                     // 启动配置是否有错
  CUDA_CHECK(cudaDeviceSynchronize());   // 等待执行完，并报告执行期间的错误

  // 4. 拷回结果并校验
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  check_close(h_c.data(), ref.data(), n);

  // 5. 计时：读两个数组、写一个数组
  float ms = time_ms([&] { vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n); });
  std::printf("vector_add: %.3f ms, %.1f GB/s\n", ms, gbps(3.0 * bytes, ms));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}
