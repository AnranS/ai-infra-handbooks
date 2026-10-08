// cluster_sum.cu —— 两个 block 组成一个集群，block 0 通过 DSMEM 读取 block 1 的共享内存
// 编译：nvcc -O3 -arch=sm_90 cluster_sum.cu -o cluster_sum
#include "common.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

constexpr int kThreads = 256;

__global__ void __cluster_dims__(2, 1, 1) pair_sum(const float* __restrict__ in, float* __restrict__ out, int per_block) {
  __shared__ float partial[kThreads];
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank();   // 本 block 在集群中的编号：0 或 1

  // 每个 block 先求自己那一段的和
  const float* src = in + static_cast<size_t>(blockIdx.x) * per_block;
  float v = 0.f;
  for (int i = threadIdx.x; i < per_block; i += blockDim.x) v += src[i];
  partial[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }

  cluster.sync();   // 两个 block 的 partial[0] 都已就绪，并且对集群内可见
  if (rank == 0 && threadIdx.x == 0) {
    float* remote = cluster.map_shared_rank(&partial[0], 1);   // 映射到 block 1 的共享内存
    out[blockIdx.x / 2] = partial[0] + *remote;
  }
  cluster.sync();   // block 1 必须等 block 0 读完才能退出，否则它的共享内存会被释放
}

int main() {
  require_sm(9, 0);
  const int clusters = 64, per_block = 10000;
  const int blocks = clusters * 2;
  std::vector<float> h(static_cast<size_t>(blocks) * per_block), ref(clusters), got(clusters);
  fill_random(h, 5);
  for (int c = 0; c < clusters; ++c) {
    double s = 0;
    for (int i = 0; i < 2 * per_block; ++i) s += h[static_cast<size_t>(c) * 2 * per_block + i];
    ref[c] = static_cast<float>(s);
  }
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, h.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, clusters * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
  pair_sum<<<blocks, kThreads>>>(d_in, d_out, per_block);   // grid 大小必须是集群大小的整数倍
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, clusters * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(got.data(), ref.data(), clusters, 1e-4f, 1e-3f);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
