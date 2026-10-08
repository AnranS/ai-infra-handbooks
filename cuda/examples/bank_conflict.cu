// bank_conflict.cu —— 观察共享内存 bank 冲突的代价
// 编译：nvcc -O3 -arch=sm_75 bank_conflict.cu -o bank_conflict
#include "common.cuh"

constexpr int kWords = 32 * 33;

// 每个线程反复读取 s[lane * STRIDE]
template <int STRIDE>
__global__ void smem_stride(float* out, int iters) {
  __shared__ float s[kWords];
  for (int i = threadIdx.x; i < kWords; i += blockDim.x) s[i] = static_cast<float>(i);
  __syncthreads();

  volatile float* vs = s;  // volatile：强制每次循环都真的去读共享内存
  const int idx = (threadIdx.x % 32) * STRIDE;
  float acc = 0.f;
  for (int k = 0; k < iters; ++k) acc += vs[idx];
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <int STRIDE>
bool run(float* d_out, int blocks, int threads, int iters) {
  smem_stride<STRIDE><<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK_LAST();
  const int n = blocks * threads;
  std::vector<float> h(n), ref(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int t = 0; t < n; ++t) ref[t] = static_cast<float>(iters) * ((t % threads) % 32 * STRIDE);
  std::printf("stride %2d: ", STRIDE);
  bool ok = check_close(h.data(), ref.data(), n, 1e-6f, 0.f);
  float ms = time_ms([&] { smem_stride<STRIDE><<<blocks, threads>>>(d_out, iters); });
  std::printf("           time %.3f ms\n", ms);
  return ok;
}

int main() {
  const int blocks = sm_count() * 4, threads = 256, iters = 4096;
  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
  bool ok = run<1>(d_out, blocks, threads, iters);   // 无冲突
  ok &= run<2>(d_out, blocks, threads, iters);       // 2 路冲突
  ok &= run<4>(d_out, blocks, threads, iters);       // 4 路冲突
  ok &= run<32>(d_out, blocks, threads, iters);      // 32 路冲突
  ok &= run<33>(d_out, blocks, threads, iters);      // 填充后无冲突
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
