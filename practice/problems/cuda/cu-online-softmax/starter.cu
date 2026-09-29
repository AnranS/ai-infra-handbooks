// 行 softmax：每行一个 block（128 线程），online softmax 只读两遍输入。
#include <cfloat>
#define BLOCK 128

__global__ void softmax_rows(const float* x, float* y, int rows, int cols) {
  // TODO：1) 每个线程 online 地扫描自己的元素得到 (m, s)；2) block 内合并；3) 写 y = exp(x - m) / s
}

void launch_softmax(const float* x, float* y, int rows, int cols) {
  softmax_rows<<<rows, BLOCK>>>(x, y, rows, cols);
}
