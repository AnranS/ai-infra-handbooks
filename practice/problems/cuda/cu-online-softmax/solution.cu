#include <cfloat>
#define BLOCK 128

__device__ __forceinline__ void combine(float& m, float& s, float m2, float s2) {
  float mn = fmaxf(m, m2);
  if (mn == -FLT_MAX) return;
  s = s * __expf(m - mn) + s2 * __expf(m2 - mn);
  m = mn;
}

__global__ void softmax_rows(const float* x, float* y, int rows, int cols) {
  __shared__ float sm[32], ss[32];
  const float* xr = x + (size_t)blockIdx.x * cols;
  float* yr = y + (size_t)blockIdx.x * cols;
  int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  float m = -FLT_MAX, s = 0.f;
  for (int c = tid; c < cols; c += BLOCK) combine(m, s, xr[c], 1.f);
  for (int off = 16; off > 0; off >>= 1) {
    float m2 = __shfl_down_sync(0xffffffffu, m, off);
    float s2 = __shfl_down_sync(0xffffffffu, s, off);
    combine(m, s, m2, s2);
  }
  if (lane == 0) sm[warp] = m, ss[warp] = s;
  __syncthreads();
  if (warp == 0) {
    m = lane < BLOCK / 32 ? sm[lane] : -FLT_MAX;
    s = lane < BLOCK / 32 ? ss[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) {
      float m2 = __shfl_down_sync(0xffffffffu, m, off);
      float s2 = __shfl_down_sync(0xffffffffu, s, off);
      combine(m, s, m2, s2);
    }
  }
  __syncthreads();
  if (tid == 0) sm[0] = m, ss[0] = s;
  __syncthreads();
  m = sm[0];
  float inv = 1.f / ss[0];
  for (int c = tid; c < cols; c += BLOCK) yr[c] = __expf(xr[c] - m) * inv;
}

void launch_softmax(const float* x, float* y, int rows, int cols) {
  softmax_rows<<<rows, BLOCK>>>(x, y, rows, cols);
}
