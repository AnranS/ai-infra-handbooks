#define BLOCK 256

__global__ void reduce_kernel(const float* x, float* partial, int n) {
  __shared__ float s[BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * 2 * BLOCK + tid;
  float v = i < n ? x[i] : 0.f;
  if (i + BLOCK < n) v += x[i + BLOCK];
  s[tid] = v;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = s[0];
}

float reduce_sum(const float* d_x, int n) {
  if (n == 0) return 0.f;
  const float* cur = d_x;
  float* bufs[2] = {nullptr, nullptr};
  int grid0 = (n + 2 * BLOCK - 1) / (2 * BLOCK);
  cudaMalloc(&bufs[0], grid0 * sizeof(float));
  cudaMalloc(&bufs[1], grid0 * sizeof(float));
  int which = 0;
  while (n > 1) {
    int grid = (n + 2 * BLOCK - 1) / (2 * BLOCK);
    reduce_kernel<<<grid, BLOCK>>>(cur, bufs[which], n);
    cur = bufs[which];
    which ^= 1;
    n = grid;
  }
  float result;
  cudaMemcpy(&result, cur, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(bufs[0]);
  cudaFree(bufs[1]);
  return result;
}
