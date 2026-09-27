// Emulator stand-in for the parts of CUB used in the handbook (computed on the CPU).
#pragma once
#include <cuda_runtime.h>
namespace cub {
struct DeviceReduce {
  template <class In, class Out>
  static cudaError_t Sum(void* temp, size_t& bytes, In in, Out out, int n, cudaStream_t = nullptr) {
    if (temp == nullptr) { bytes = 16; return cudaSuccess; }
    double s = 0;
    for (int i = 0; i < n; ++i) s += in[i];
    *out = static_cast<float>(s);
    return cudaSuccess;
  }
};
}  // namespace cub
