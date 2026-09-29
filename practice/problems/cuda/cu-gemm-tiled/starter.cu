// C = A @ B，行主序，A: M×K，B: K×N。共享内存分块，TILE × TILE 的 block 计算 C 的一个块。
#define TILE 16

__global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  // TODO
}

void launch_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_tiled<<<grid, block>>>(A, B, C, M, N, K);
}
