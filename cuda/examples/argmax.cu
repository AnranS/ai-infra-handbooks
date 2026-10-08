// argmax.cu —— 用 warp shuffle 同时归约值和下标
// 编译：nvcc -O3 -arch=sm_75 argmax.cu -o argmax
#include "common.cuh"

struct Pair {
  float v;
  int i;
};

__device__ __forceinline__ Pair better(Pair a, Pair b) {
  return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
}

__device__ __forceinline__ Pair warp_argmax(Pair p) {
  for (int offset = 16; offset > 0; offset /= 2) {
    Pair o{__shfl_down_sync(0xffffffff, p.v, offset), __shfl_down_sync(0xffffffff, p.i, offset)};
    p = better(p, o);
  }
  return p;
}

__global__ void argmax_kernel(const float* __restrict__ x, int n, float* bv, int* bi) {
  Pair p{-INFINITY, 0x7fffffff};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    p = better(p, Pair{x[i], i});
  __shared__ Pair ws[32];
  p = warp_argmax(p);
  if (threadIdx.x % 32 == 0) ws[threadIdx.x / 32] = p;
  __syncthreads();
  if (threadIdx.x < 32) {
    p = threadIdx.x < blockDim.x / 32 ? ws[threadIdx.x] : Pair{-INFINITY, 0x7fffffff};
    p = warp_argmax(p);
    if (threadIdx.x == 0) {
      bv[blockIdx.x] = p.v;
      bi[blockIdx.x] = p.i;
    }
  }
}

int main() {
  const int n = 10'000'019;
  std::vector<float> h(n);
  fill_random(h, 5);
  h[7'654'321] = 3.f;   // 唯一的最大值
  const int blocks = sm_count() * 4, threads = 256;
  float *dx, *dbv;
  int* dbi;
  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dbv, blocks * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dbi, blocks * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dx, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  argmax_kernel<<<blocks, threads>>>(dx, n, dbv, dbi);
  CUDA_CHECK_LAST();
  std::vector<float> bv(blocks);
  std::vector<int> bi(blocks);
  CUDA_CHECK(cudaMemcpy(bv.data(), dbv, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(bi.data(), dbi, blocks * sizeof(int), cudaMemcpyDeviceToHost));
  int best = 0;
  for (int b = 1; b < blocks; ++b)
    if (bv[b] > bv[best] || (bv[b] == bv[best] && bi[b] < bi[best])) best = b;
  bool ok = bi[best] == 7'654'321;
  std::printf("argmax = %d (%s)\n", bi[best], ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
