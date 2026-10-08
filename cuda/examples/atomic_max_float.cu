// atomic_max_float.cu —— 用 atomicCAS 实现浮点原子最大值
// 编译：nvcc -O3 -arch=sm_75 atomic_max_float.cu -o atomic_max_float
#include "common.cuh"

__device__ float atomicMaxFloat(float* addr, float val) {
  int* p = reinterpret_cast<int*>(addr);
  int old = *p;
  while (__int_as_float(old) < val) {
    int assumed = old;
    old = atomicCAS(p, assumed, __float_as_int(val));   // 成功时返回 assumed
    if (old == assumed) break;
  }
  return __int_as_float(old);
}

__global__ void max_kernel(const float* x, int n, float* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    atomicMaxFloat(out, x[i]);
}

int main() {
  const int n = 1 << 22;
  std::vector<float> h(n);
  fill_random(h, 11, -100.f, 100.f);
  float ref = *std::max_element(h.begin(), h.end());

  float *d_x, *d_out;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  float init = -INFINITY;
  CUDA_CHECK(cudaMemcpy(d_out, &init, sizeof(float), cudaMemcpyHostToDevice));
  max_kernel<<<sm_count() * 4, 256>>>(d_x, n, d_out);
  CUDA_CHECK_LAST();
  float got;
  CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  return check_close(&got, &ref, 1, 0.f, 0.f) ? 0 : 1;
}
