// compact.cu —— 用 ballot + popc + warp 聚合原子操作做流压缩（不保序）
// 编译：nvcc -O3 -arch=sm_75 compact.cu -o compact
#include "common.cuh"

__global__ void compact_positive(const float* __restrict__ in, float* __restrict__ out,
                                 int* __restrict__ count, int n) {
  const int lane = threadIdx.x % 32;
  // 整个 warp 一起循环，保证 __ballot_sync 时 warp 内线程都在
  for (int base = blockIdx.x * blockDim.x; base < n; base += blockDim.x * gridDim.x) {
    const int i = base + threadIdx.x;
    const float v = i < n ? in[i] : 0.f;
    const bool keep = i < n && v > 0.f;
    const unsigned mask = __ballot_sync(0xffffffff, keep);
    const int rank = __popc(mask & ((1u << lane) - 1));
    int start = 0;
    if (lane == 0 && mask) start = atomicAdd(count, __popc(mask));
    start = __shfl_sync(0xffffffff, start, 0);
    if (keep) out[start + rank] = v;
  }
}

int main() {
  const int n = 1 << 22;
  std::vector<float> h(n);
  fill_random(h, 3);
  std::vector<float> ref;
  for (float v : h) if (v > 0.f) ref.push_back(v);

  float *d_in, *d_out;
  int* d_count;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  compact_positive<<<sm_count() * 8, 256>>>(d_in, d_out, d_count, n);
  CUDA_CHECK_LAST();
  int count = 0;
  CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<float> got(count);
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));
  // 顺序不保证，排序后比较
  std::sort(got.begin(), got.end());
  std::sort(ref.begin(), ref.end());
  bool ok = got == ref;
  std::printf("kept %d of %d: %s\n", count, n, ok ? "PASS" : "FAIL");
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_count));
  return ok ? 0 : 1;
}
