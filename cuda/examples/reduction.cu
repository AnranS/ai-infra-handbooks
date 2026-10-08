// reduction.cu —— 归约优化的七个版本，逐个验证正确性并测量有效带宽
// 编译：nvcc -O3 -arch=sm_75 reduction.cu -o reduction
#include "common.cuh"
#include <cub/cub.cuh>

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

// v0：交错寻址 + 取模判断，warp 内严重发散
__global__ void reduce_v0(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[kThreads];
  unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// v1：连续线程工作，没有发散，但跨步访问导致 bank 冲突
__global__ void reduce_v1(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[kThreads];
  unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
    unsigned idx = 2 * stride * tid;
    if (idx < blockDim.x) s[idx] += s[idx + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// v2：顺序寻址，无发散、无 bank 冲突
__global__ void reduce_v2(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[kThreads];
  unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = i < n ? in[i] : 0.f;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// v3：每个 block 处理 2 * blockDim 个元素，加载时先加一次
__global__ void reduce_v3(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[kThreads];
  unsigned tid = threadIdx.x, i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float v = i < n ? in[i] : 0.f;
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  s[tid] = v;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// v4：v3 + 最后 64 -> 1 用 warp shuffle（要求 blockDim >= 64）
__global__ void reduce_v4(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float s[kThreads];
  unsigned tid = threadIdx.x, i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float v = i < n ? in[i] : 0.f;
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  s[tid] = v;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid < 32) {
    v = warp_reduce_sum(s[tid] + s[tid + 32]);
    if (tid == 0) out[blockIdx.x] = v;
  }
}

// v5：grid-stride + float4 + 两级 shuffle 归约
__global__ void reduce_v5(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = blockDim.x * gridDim.x;
  float v = 0.f;
  const int n4 = n / 4;
  const float4* in4 = reinterpret_cast<const float4*>(in);
  for (int i = gtid; i < n4; i += nthreads) {
    float4 t = in4[i];
    v += (t.x + t.y) + (t.z + t.w);
  }
  for (int i = n4 * 4 + gtid; i < n; i += nthreads) v += in[i];   // 不足 4 个的尾部

  __shared__ float warp_sums[32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  v = warp_reduce_sum(v);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < blockDim.x / 32 ? warp_sums[lane] : 0.f;
    v = warp_reduce_sum(v);
    if (lane == 0) out[blockIdx.x] = v;
  }
}

struct Version {
  const char* name;
  void (*kernel)(const float*, float*, int);
  int elems_per_block;   // 0 表示用固定的 grid 大小（grid-stride）
};

int main() {
  const int n = (1 << 26) + 3;   // 约 6700 万个元素，故意不是 4 的倍数
  std::vector<float> h(n);
  fill_random(h, 2024, 0.f, 1.f);
  double ref = 0.0;
  for (float v : h) ref += v;

  float *d_in, *d_partial, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partial, ((n + kThreads - 1) / kThreads) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const Version versions[] = {
      {"v0 interleaved+divergent", reduce_v0, kThreads},
      {"v1 interleaved+conflicts", reduce_v1, kThreads},
      {"v2 sequential", reduce_v2, kThreads},
      {"v3 first add on load", reduce_v3, 2 * kThreads},
      {"v4 warp shuffle tail", reduce_v4, 2 * kThreads},
      {"v5 grid-stride+float4", reduce_v5, 0},
  };
  const double bytes = static_cast<double>(n) * sizeof(float);
  bool ok = true;
  std::printf("%-28s %10s %10s  check\n", "version", "time(ms)", "GB/s");
  for (const auto& ver : versions) {
    const int blocks = ver.elems_per_block ? (n + ver.elems_per_block - 1) / ver.elems_per_block : sm_count() * 8;
    ver.kernel<<<blocks, kThreads>>>(d_in, d_partial, n);
    CUDA_CHECK_LAST();
    std::vector<float> partial(blocks);
    CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (float p : partial) sum += p;   // 第二步：部分和很少，这里在 CPU 上合并
    bool good = std::fabs(sum - ref) <= 1e-5 * ref;
    ok &= good;
    float ms = time_ms([&] { ver.kernel<<<blocks, kThreads>>>(d_in, d_partial, n); });
    std::printf("%-28s %10.3f %10.1f  %s\n", ver.name, ms, gbps(bytes, ms), good ? "PASS" : "FAIL");
  }

  // v6：CUB 设备级归约
  size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, d_in, d_out, n));
  void* d_temp;
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
  CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n));
  float cub_sum = 0.f;
  CUDA_CHECK(cudaMemcpy(&cub_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  bool good = std::fabs(cub_sum - ref) <= 1e-4 * ref;
  ok &= good;
  float ms = time_ms([&] { cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n); });
  std::printf("%-28s %10.3f %10.1f  %s\n", "v6 cub::DeviceReduce", ms, gbps(bytes, ms), good ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_partial));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_temp));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
