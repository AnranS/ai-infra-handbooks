// access_pattern.cu —— 测量跨步访问和非对齐访问对带宽的影响
// 编译：nvcc -O3 -arch=sm_75 access_pattern.cu -o access_pattern
#include "common.cuh"

__global__ void strided_copy(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[static_cast<size_t>(i) * stride];
}

__global__ void offset_copy(const float* __restrict__ in, float* __restrict__ out, int n, int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i + offset];
}

int main() {
  const int n = 1 << 22;             // 每次拷贝 4M 个有效元素
  const int max_stride = 32;
  const size_t in_elems = static_cast<size_t>(n) * max_stride + 64;
  std::vector<float> h_in(in_elems);
  for (size_t i = 0; i < in_elems; ++i) h_in[i] = static_cast<float>(i % 1000);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, in_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256, blocks = (n + threads - 1) / threads;
  std::vector<float> h_out(n), ref(n);
  bool ok = true;

  std::printf("stride  time(ms)  useful GB/s\n");
  for (int stride : {1, 2, 4, 8, 16, 32}) {
    strided_copy<<<blocks, threads>>>(d_in, d_out, n, stride);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) ref[i] = h_in[static_cast<size_t>(i) * stride];
    ok &= check_close(h_out.data(), ref.data(), n, 0.f, 0.f);
    float ms = time_ms([&] { strided_copy<<<blocks, threads>>>(d_in, d_out, n, stride); });
    // 只统计"有用"的字节：读 n 个、写 n 个 float
    std::printf("%6d  %8.3f  %10.1f\n", stride, ms, gbps(2.0 * n * sizeof(float), ms));
  }

  std::printf("\noffset  time(ms)  GB/s\n");
  for (int offset : {0, 1, 2, 4, 8, 16, 32}) {
    float ms = time_ms([&] { offset_copy<<<blocks, threads>>>(d_in, d_out, n, offset); });
    std::printf("%6d  %8.3f  %10.1f\n", offset, ms, gbps(2.0 * n * sizeof(float), ms));
  }
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
