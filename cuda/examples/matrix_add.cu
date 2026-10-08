// matrix_add.cu —— 二维线程组织
// 编译：nvcc -O3 -arch=sm_75 matrix_add.cu -o matrix_add
#include "common.cuh"

__global__ void matrix_add(const float* A, const float* B, float* C, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // x 对应列：相邻线程访问相邻地址
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    size_t idx = static_cast<size_t>(row) * cols + col;
    C[idx] = A[idx] + B[idx];
  }
}

int main() {
  const int rows = 3000, cols = 5000;  // 故意取不能整除的大小
  const size_t n = static_cast<size_t>(rows) * cols, bytes = n * sizeof(float);
  std::vector<float> hA(n), hB(n), hC(n), ref(n);
  fill_random(hA, 1);
  fill_random(hB, 2);
  for (size_t i = 0; i < n; ++i) ref[i] = hA[i] + hB[i];

  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, bytes));
  CUDA_CHECK(cudaMalloc(&B, bytes));
  CUDA_CHECK(cudaMalloc(&C, bytes));
  CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(32, 8);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  matrix_add<<<grid, block>>>(A, B, C, rows, cols);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
  check_close(hC.data(), ref.data(), n);

  float ms = time_ms([&] { matrix_add<<<grid, block>>>(A, B, C, rows, cols); });
  std::printf("matrix_add: %.3f ms, %.1f GB/s\n", ms, gbps(3.0 * bytes, ms));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
