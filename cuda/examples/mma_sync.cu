// mma_sync.cu —— 直接使用 PTX mma.sync.m16n8k16，手动按布局装载寄存器
// 编译：nvcc -O3 -arch=sm_80 mma_sync.cu -o mma_sync
#include "common.cuh"
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ uint32_t pack_half2(half lo, half hi) {
  __half2 h = __halves2half2(lo, hi);   // lo 在低 16 位，对应编号较小的元素
  return *reinterpret_cast<uint32_t*>(&h);
}

// A: 16x16 行主序，B: 16x8 行主序，D: 16x8 行主序（FP32）
__global__ void mma_16x8x16(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ D) {
  const int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  auto a = [&](int r, int c) { return A[r * 16 + c]; };
  auto b = [&](int r, int c) { return B[r * 8 + c]; };
  uint32_t ra[4], rb[2];
  ra[0] = pack_half2(a(g, 2 * t), a(g, 2 * t + 1));
  ra[1] = pack_half2(a(g + 8, 2 * t), a(g + 8, 2 * t + 1));
  ra[2] = pack_half2(a(g, 2 * t + 8), a(g, 2 * t + 9));
  ra[3] = pack_half2(a(g + 8, 2 * t + 8), a(g + 8, 2 * t + 9));
  rb[0] = pack_half2(b(2 * t, g), b(2 * t + 1, g));
  rb[1] = pack_half2(b(2 * t + 8, g), b(2 * t + 9, g));
  float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  D[g * 8 + 2 * t] = d[0];
  D[g * 8 + 2 * t + 1] = d[1];
  D[(g + 8) * 8 + 2 * t] = d[2];
  D[(g + 8) * 8 + 2 * t + 1] = d[3];
}

int main() {
  require_sm(8, 0);
  std::vector<float> fa(16 * 16), fb(16 * 8), ref(16 * 8, 0.f), got(16 * 8);
  fill_random(fa, 1);
  fill_random(fb, 2);
  std::vector<half> ha(fa.size()), hb(fb.size());
  for (size_t i = 0; i < fa.size(); ++i) { ha[i] = __float2half(fa[i]); fa[i] = __half2float(ha[i]); }
  for (size_t i = 0; i < fb.size(); ++i) { hb[i] = __float2half(fb[i]); fb[i] = __half2float(hb[i]); }
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 16; ++k)
      for (int j = 0; j < 8; ++j) ref[i * 8 + j] += fa[i * 16 + k] * fb[k * 8 + j];

  half *dA, *dB;
  float* dD;
  CUDA_CHECK(cudaMalloc(&dA, ha.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB, hb.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dD, got.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, ha.data(), ha.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hb.data(), hb.size() * sizeof(half), cudaMemcpyHostToDevice));
  mma_16x8x16<<<1, 32>>>(dA, dB, dD);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), dD, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = check_close(got.data(), ref.data(), got.size(), 1e-3f, 1e-3f);
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dD));
  return ok ? 0 : 1;
}
