// occupancy.cu —— 用 Occupancy API 计算不同配置下的占用率
// 编译：nvcc -O3 -arch=sm_75 occupancy.cu -o occupancy
#include "common.cuh"

__global__ void light_kernel(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * 2.f + 1.f;
}

// 每个 block 使用 dynamic shared memory，模拟共享内存限制
__global__ void smem_kernel(float* x, int n) {
  extern __shared__ float buf[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  buf[threadIdx.x] = (i < n) ? x[i] : 0.f;
  __syncthreads();
  if (i < n) x[i] = buf[blockDim.x - 1 - threadIdx.x];
}

template <typename K>
void report(const char* name, K kernel, int block, size_t smem) {
  int dev = 0, blocks_per_sm = 0;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block, smem));
  float occ = static_cast<float>(blocks_per_sm * block) / p.maxThreadsPerMultiProcessor;
  std::printf("%-12s block=%4d smem=%6zu B -> %2d blocks/SM, occupancy %5.1f%%\n",
              name, block, smem, blocks_per_sm, occ * 100);
}

int main() {
  for (int block : {32, 64, 128, 256, 512, 1024}) report("light", light_kernel, block, 0);
  for (size_t smem : {0, 16 * 1024, 32 * 1024, 48 * 1024}) report("smem", smem_kernel, 256, smem);

  // 让运行时推荐一个 block 大小（以占用率最大化为目标）
  int min_grid = 0, best_block = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block, light_kernel, 0, 0));
  std::printf("suggested block size for light_kernel: %d (min grid %d)\n", best_block, min_grid);
  return 0;
}
