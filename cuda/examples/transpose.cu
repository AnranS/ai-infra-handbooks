// transpose.cu —— 矩阵转置：朴素 → 共享内存 → 消除 bank 冲突，并与拷贝对比
// 编译：nvcc -O3 -arch=sm_75 transpose.cu -o transpose
#include "common.cuh"

constexpr int TILE = 32;
constexpr int ROWS_PER_ITER = 8;   // block 为 32 x 8，每个线程处理 TILE / 8 = 4 个元素

// 上限参照：同样的访问形状，只是不转置
__global__ void copy_tile(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < cols && y + j < rows) out[(y + j) * cols + x] = in[(y + j) * cols + x];
}

// 朴素：读合并，写不合并
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < cols && y + j < rows) out[x * rows + (y + j)] = in[(y + j) * cols + x];
}

// 共享内存中转，tile 宽 32：按列读共享内存时有 32 路 bank 冲突
// 共享内存中转，tile 宽 33：填充一列消除冲突
template <int PAD>
__global__ void transpose_smem(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + PAD];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
  __syncthreads();

  // 输出块的位置：块坐标对调
  x = blockIdx.y * TILE + threadIdx.x;   // 输出的列 = 输入的行
  y = blockIdx.x * TILE + threadIdx.y;   // 输出的行 = 输入的列
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < rows && y + j < cols) out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main() {
  const int rows = 4096 + 17, cols = 8192 + 5;   // 故意取非 32 的倍数，检验边界处理
  const size_t n = static_cast<size_t>(rows) * cols, bytes = n * sizeof(float);
  std::vector<float> h(n), ref(n), got(n);
  fill_random(h, 9);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) ref[static_cast<size_t>(c) * rows + r] = h[static_cast<size_t>(r) * cols + c];

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, ROWS_PER_ITER);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  bool ok = true;

  auto bench = [&](const char* name, auto kernel, bool check) {
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    CUDA_CHECK_LAST();
    std::printf("%-22s ", name);
    if (check) {
      CUDA_CHECK(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost));
      ok &= check_close(got.data(), ref.data(), n, 0.f, 0.f);
      std::printf("%-22s ", "");
    }
    float ms = time_ms([&] { kernel<<<grid, block>>>(d_in, d_out, rows, cols); });
    std::printf("%.3f ms, %.1f GB/s\n", ms, gbps(2.0 * bytes, ms));
  };

  bench("copy (upper bound)", copy_tile, false);
  bench("naive", transpose_naive, true);
  bench("smem [32][32]", transpose_smem<0>, true);
  bench("smem [32][33]", transpose_smem<1>, true);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
