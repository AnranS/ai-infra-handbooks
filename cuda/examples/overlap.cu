// overlap.cu —— 锁页内存 + 多流：把拷贝和计算重叠起来
// 编译：nvcc -O3 -arch=sm_75 overlap.cu -o overlap
#include "common.cuh"

__global__ void heavy(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    for (int k = 0; k < 64; ++k) v = v * 0.999f + 0.001f;   // 人为增加计算量，使计算与拷贝耗时相当
    x[i] = v;
  }
}

float ref_value(float v) {
  for (int k = 0; k < 64; ++k) v = v * 0.999f + 0.001f;
  return v;
}

int main() {
  const int n = 1 << 26, chunks = 8, chunk = n / chunks;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float *h, *d;
  CUDA_CHECK(cudaMallocHost(&h, bytes));   // 锁页内存：异步拷贝的前提
  CUDA_CHECK(cudaMalloc(&d, bytes));
  auto init = [&] { for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i % 100) / 100.f; };

  // 串行版本：整块拷入 → 计算 → 整块拷出
  init();
  GpuTimer t;
  t.start();
  CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
  heavy<<<(n + 255) / 256, 256>>>(d, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
  float serial_ms = t.stop();

  // 流水线版本：分块，每块在自己的流里依次执行拷入、计算、拷出
  init();
  std::vector<cudaStream_t> streams(4);
  for (auto& s : streams) CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  t.start();   // 在默认流上记录；计时范围内用 cudaDeviceSynchronize 保证覆盖所有流
  for (int c = 0; c < chunks; ++c) {
    cudaStream_t s = streams[c % streams.size()];
    const size_t off = static_cast<size_t>(c) * chunk;
    CUDA_CHECK(cudaMemcpyAsync(d + off, h + off, chunk * sizeof(float), cudaMemcpyHostToDevice, s));
    heavy<<<(chunk + 255) / 256, 256, 0, s>>>(d + off, chunk);
    CUDA_CHECK(cudaMemcpyAsync(h + off, d + off, chunk * sizeof(float), cudaMemcpyDeviceToHost, s));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  float pipelined_ms = t.stop();

  size_t bad = 0;
  for (int i = 0; i < n; ++i) bad += std::fabs(h[i] - ref_value(static_cast<float>(i % 100) / 100.f)) > 1e-5f;
  std::printf("%s (%zu mismatches)\nserial   : %.3f ms\npipelined: %.3f ms\n", bad ? "FAIL" : "PASS", bad,
              serial_ms, pipelined_ms);
  for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFreeHost(h));
  CUDA_CHECK(cudaFree(d));
  return bad ? 1 : 0;
}
