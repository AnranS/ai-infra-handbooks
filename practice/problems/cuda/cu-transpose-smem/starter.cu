// 共享内存矩阵转置：a 是 rows×cols（行主序），b = aᵀ 是 cols×rows。
// launch_transpose 的启动配置已经写好：block = (32, 8)，每个 block 负责一个 32×32 的块。
#define TILE 32
#define ROWS_PER_PASS 8

__global__ void transpose_kernel(const float* a, float* b, int rows, int cols) {
  // TODO：用 __shared__ float tile[TILE][TILE + 1] 实现合并读、合并写的转置。
  // 读阶段：线程 (tx, ty) 读 a[(by*32 + ty + j) * cols + bx*32 + tx]，写进 tile[ty + j][tx]；
  // __syncthreads() 之后交换块坐标，读 tile[tx][ty + j]，合并地写 b。
}

void launch_transpose(const float* a, float* b, int rows, int cols) {
  dim3 block(TILE, ROWS_PER_PASS);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  transpose_kernel<<<grid, block>>>(a, b, rows, cols);
}
