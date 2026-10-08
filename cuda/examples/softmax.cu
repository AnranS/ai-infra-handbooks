// softmax.cu —— 行 softmax 的三种实现：block 三遍、warp 一行（寄存器缓存）、block online 两遍
// 编译：nvcc -O3 -arch=sm_75 softmax.cu -o softmax
#include "common.cuh"

__device__ __forceinline__ float warp_max(float v) {
  for (int m = 16; m > 0; m /= 2) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, m));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int m = 16; m > 0; m /= 2) v += __shfl_xor_sync(0xffffffff, v, m);
  return v;
}

// block 内归约，所有线程都拿到结果。op: 0 = sum, 1 = max
template <int OP>
__device__ float block_allreduce(float v) {
  __shared__ float buf[32];
  __shared__ float result;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
  v = OP ? warp_max(v) : warp_sum(v);
  if (lane == 0) buf[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < nwarps ? buf[lane] : (OP ? -INFINITY : 0.f);
    v = OP ? warp_max(v) : warp_sum(v);
    if (lane == 0) result = v;
  }
  __syncthreads();
  v = result;
  __syncthreads();   // 防止下一次调用覆盖 buf/result 时还有线程没读完
  return v;
}

// v1：一个 block 一行，三次遍历全局内存（max、sum、写出）
__global__ void softmax_block_3pass(const float* __restrict__ x, float* __restrict__ y, int cols) {
  const float* row = x + static_cast<size_t>(blockIdx.x) * cols;
  float* out = y + static_cast<size_t>(blockIdx.x) * cols;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, row[i]);
  m = block_allreduce<1>(m);
  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) s += __expf(row[i] - m);
  s = block_allreduce<0>(s);
  const float inv = 1.f / s;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = __expf(row[i] - m) * inv;
}

// v2：一个 warp 一行，整行缓存在寄存器里，显存只读一次、写一次。要求 cols <= COLS
template <int COLS>
__global__ void softmax_warp(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
  constexpr int kPerLane = (COLS + 31) / 32;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= rows) return;   // 整个 warp 一起退出，不影响其他 warp
  const float* in = x + static_cast<size_t>(row) * cols;
  float v[kPerLane];
  float m = -INFINITY;
#pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    int c = k * 32 + lane;                 // 相邻 lane 读相邻元素：合并访问
    v[k] = c < cols ? in[c] : -INFINITY;
    m = fmaxf(m, v[k]);
  }
  m = warp_max(m);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    v[k] = __expf(v[k] - m);               // exp(-inf) = 0，越界位置不影响求和
    s += v[k];
  }
  s = warp_sum(s);
  const float inv = 1.f / s;
  float* out = y + static_cast<size_t>(row) * cols;
#pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    int c = k * 32 + lane;
    if (c < cols) out[c] = v[k] * inv;
  }
}

// online softmax 的 (m, d) 合并
struct MD {
  float m, d;
};
__device__ __forceinline__ MD md_combine(MD a, MD b) {
  float m = fmaxf(a.m, b.m);
  // 两边都是 -inf（还没有元素）时避免出现 inf - inf = nan
  float da = a.m == -INFINITY ? 0.f : a.d * __expf(a.m - m);
  float db = b.m == -INFINITY ? 0.f : b.d * __expf(b.m - m);
  return {m, da + db};
}
__device__ MD warp_md(MD v) {
  for (int k = 16; k > 0; k /= 2) {
    MD o{__shfl_xor_sync(0xffffffff, v.m, k), __shfl_xor_sync(0xffffffff, v.d, k)};
    v = md_combine(v, o);
  }
  return v;
}

// v3：一个 block 一行，online softmax：一次遍历求 (m, d)，一次遍历写出
__global__ void softmax_block_online(const float* __restrict__ x, float* __restrict__ y, int cols) {
  const float* row = x + static_cast<size_t>(blockIdx.x) * cols;
  float* out = y + static_cast<size_t>(blockIdx.x) * cols;
  MD md{-INFINITY, 0.f};
  for (int i = threadIdx.x; i < cols; i += blockDim.x) md = md_combine(md, MD{row[i], 1.f});
  __shared__ MD buf[32];
  __shared__ MD total;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  md = warp_md(md);
  if (lane == 0) buf[warp] = md;
  __syncthreads();
  if (warp == 0) {
    md = lane < blockDim.x / 32 ? buf[lane] : MD{-INFINITY, 0.f};
    md = warp_md(md);
    if (lane == 0) total = md;
  }
  __syncthreads();
  const float m = total.m, inv = 1.f / total.d;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = __expf(row[i] - m) * inv;
}

void softmax_cpu(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* in = x + static_cast<size_t>(r) * cols;
    float* out = y + static_cast<size_t>(r) * cols;
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = std::max(m, in[c]);
    double s = 0;
    for (int c = 0; c < cols; ++c) s += std::exp(double(in[c]) - m);
    for (int c = 0; c < cols; ++c) out[c] = static_cast<float>(std::exp(double(in[c]) - m) / s);
  }
}

int main() {
  bool ok = true;
  // 两种形状：注意力分数（行短、行数多）和词表 logits（行长、行数少）
  struct Shape { int rows, cols; } shapes[] = {{4096, 1000}, {64, 50000}};
  for (const Shape& sh : shapes) {
    const int rows = sh.rows, cols = sh.cols;
    const size_t n = static_cast<size_t>(rows) * cols, bytes = n * sizeof(float);
    std::vector<float> hx(n), ref(n), got(n);
    fill_random(hx, rows, -20.f, 20.f);
    hx[5] = 90.f;   // 一个很大的值：没有减最大值的实现会溢出
    softmax_cpu(hx.data(), ref.data(), rows, cols);
    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    std::printf("rows=%d cols=%d\n", rows, cols);

    auto run = [&](const char* name, auto launch) {
      CUDA_CHECK(cudaMemset(dy, 0, bytes));
      launch();
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaMemcpy(got.data(), dy, bytes, cudaMemcpyDeviceToHost));
      std::printf("  %-22s ", name);
      ok &= check_close(got.data(), ref.data(), n, 1e-3f, 1e-6f);
      float ms = time_ms(launch);
      std::printf("  %-22s %.3f ms, %.1f GB/s (counting one read + one write)\n", "", ms, gbps(2.0 * bytes, ms));
    };
    run("block, 3 passes", [&] { softmax_block_3pass<<<rows, 256>>>(dx, dy, cols); });
    run("block, online", [&] { softmax_block_online<<<rows, 256>>>(dx, dy, cols); });
    if (cols <= 1024) {
      const int warps_per_block = 4;
      run("warp per row", [&] {
        softmax_warp<1024><<<(rows + warps_per_block - 1) / warps_per_block, 32 * warps_per_block>>>(dx, dy, rows, cols);
      });
    }
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
  }
  return ok ? 0 : 1;
}
