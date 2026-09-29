__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

void launch_vector_add(const float* a, const float* b, float* c, int n) {
  if (n == 0) return;
  int block = 256;
  int grid = (n + block - 1) / block;
  add_kernel<<<grid, block>>>(a, b, c, n);
}
