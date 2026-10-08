// tma_copy.cu —— 用 TMA 读写二维块：一个线程发起拷贝，mbarrier 按字节数等待完成
// 编译：nvcc -O3 -arch=sm_90 tma_copy.cu -o tma_copy -lcuda
#include "common.cuh"
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

constexpr int BOX_H = 16, BOX_W = 32;   // 每块 16 行 x 32 列的 int，一行 128 字节

__global__ void add_one_tma(const __grid_constant__ CUtensorMap tensor_map) {
  __shared__ alignas(128) int tile[BOX_H][BOX_W];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  const int32_t coords[2] = {static_cast<int32_t>(blockIdx.x * BOX_W),    // 最内层维度（列）在前
                             static_cast<int32_t>(blockIdx.y * BOX_H)};

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    ptx::fence_proxy_async(ptx::space_shared);   // 让 TMA（异步代理）看到初始化后的 barrier
  }
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, &tile, &tensor_map, coords,
                              cuda::device::barrier_native_handle(bar));
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile));   // 声明还要等 sizeof(tile) 个字节
  } else {
    token = bar.arrive();
  }
  bar.wait(std::move(token));

  for (int i = threadIdx.x; i < BOX_H * BOX_W; i += blockDim.x) tile[i / BOX_W][i % BOX_W] += 1;

  ptx::fence_proxy_async(ptx::space_shared);     // 让 TMA 看到普通线程对共享内存的写入
  __syncthreads();
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared, &tensor_map, coords, &tile);
    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());   // 等 TMA 读完共享内存再退出
    (&bar)->~barrier();
  }
}

int main() {
  require_sm(9, 0);
  const int rows = 1024, cols = 2048;   // 行跨度 cols * 4 字节必须是 16 的倍数
  std::vector<int> h(static_cast<size_t>(rows) * cols), got(h.size());
  for (size_t i = 0; i < h.size(); ++i) h[i] = static_cast<int>(i % 1000);
  int* d;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));

  CUtensorMap map{};
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};   // 最内层维度在前
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols) * sizeof(int)};                  // 第 1 维的跨度（字节）
  cuuint32_t box[2] = {BOX_W, BOX_H};
  cuuint32_t elem_strides[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, d, dims, strides, box, elem_strides,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::printf("cuTensorMapEncodeTiled failed: %d\n", static_cast<int>(r));
    return 1;
  }
  add_one_tma<<<dim3(cols / BOX_W, rows / BOX_H), 128>>>(map);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), d, h.size() * sizeof(int), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < h.size(); ++i) bad += got[i] != h[i] + 1;
  std::printf("TMA add-one: %s (%zu mismatches)\n", bad ? "FAIL" : "PASS", bad);
  CUDA_CHECK(cudaFree(d));
  return bad ? 1 : 0;
}
