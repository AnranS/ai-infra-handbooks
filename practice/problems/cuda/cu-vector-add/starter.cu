// c[i] = a[i] + b[i]。实现 kernel 和 launch_vector_add（计算 grid 大小并启动）。
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  // TODO
}

void launch_vector_add(const float* a, const float* b, float* c, int n) {
  int block = 256;
  // TODO：grid 大小刚好覆盖 n 个元素；n == 0 时不启动
}
