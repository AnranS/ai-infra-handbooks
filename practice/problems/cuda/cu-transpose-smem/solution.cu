#define TILE 32
#define ROWS_PER_PASS 8

__global__ void transpose_kernel(const float* a, float* b, int rows, int cols) {
  __shared__ float tile[TILE][TILE + 1];  // 多一列，错开 bank
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_PASS)
    if (y + j < rows && x < cols) tile[threadIdx.y + j][threadIdx.x] = a[(y + j) * cols + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;  // 交换块坐标
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_PASS)
    if (y + j < cols && x < rows) b[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

void launch_transpose(const float* a, float* b, int rows, int cols) {
  dim3 block(TILE, ROWS_PER_PASS);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  transpose_kernel<<<grid, block>>>(a, b, rows, cols);
}
