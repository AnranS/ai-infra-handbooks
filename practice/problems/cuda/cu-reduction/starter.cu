// 多轮归约：reduce_sum 返回 d_x[0..n) 的和。每个 block 256 线程、处理 512 个元素，
// 顺序寻址、加载时先加一次；部分和写到临时数组，再对它继续归约，直到只剩一个数。
#define BLOCK 256

__global__ void reduce_kernel(const float* x, float* partial, int n) {
  __shared__ float s[BLOCK];
  // TODO
}

float reduce_sum(const float* d_x, int n) {
  // TODO：循环启动 reduce_kernel，最后把结果拷回主机
  return 0.f;
}
