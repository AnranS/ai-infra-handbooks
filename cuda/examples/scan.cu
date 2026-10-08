// scan.cu —— warp 扫描 → block 扫描 → 三步法设备级扫描（排除式前缀和）
// 编译：nvcc -O3 -arch=sm_75 scan.cu -o scan
#include "common.cuh"

constexpr int kThreads = 256;
constexpr int kItems = 4;                       // 每个线程处理 4 个元素
constexpr int kTile = kThreads * kItems;        // 每个 block 处理 1024 个元素

__device__ __forceinline__ int warp_inclusive_scan(int v) {
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    int n = __shfl_up_sync(0xffffffff, v, offset);
    if (lane >= offset) v += n;
  }
  return v;
}

// block 内排除式扫描：返回本线程之前（不含本线程）所有线程的 v 之和，total 为整个 block 的和
__device__ int block_exclusive_scan(int v, int& total) {
  __shared__ int warp_totals[32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
  int inclusive = warp_inclusive_scan(v);
  if (lane == 31) warp_totals[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    int t = lane < nwarps ? warp_totals[lane] : 0;
    int s = warp_inclusive_scan(t);
    if (lane < nwarps) warp_totals[lane] = s - t;   // 每个 warp 的起始偏移（排除式）
    if (lane == nwarps - 1) warp_totals[31] = s;     // 暂存 block 总和
  }
  __syncthreads();
  total = warp_totals[31];
  int result = warp_totals[warp] + inclusive - v;
  __syncthreads();   // 允许同一个 kernel 里再次调用
  return result;
}

// 第 1 步：每个 block 扫描自己的 kTile 个元素，写出块内的排除式前缀和与块总和
__global__ void scan_tiles(const int* __restrict__ in, int* __restrict__ out, int* __restrict__ block_sums, int n) {
  const int base = blockIdx.x * kTile + threadIdx.x * kItems;
  int items[kItems];
  int thread_sum = 0;
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    items[k] = base + k < n ? in[base + k] : 0;
    thread_sum += items[k];
  }
  int total;
  int running = block_exclusive_scan(thread_sum, total);   // 本线程之前的所有元素之和
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    if (base + k < n) out[base + k] = running;
    running += items[k];
  }
  if (threadIdx.x == 0) block_sums[blockIdx.x] = total;
}

// 第 2 步：用一个 block 对 block_sums 做排除式扫描（按 kThreads 一段一段处理任意数量的块）
__global__ void scan_block_sums(int* block_sums, int num_blocks) {
  __shared__ int carry;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (int start = 0; start < num_blocks; start += blockDim.x) {
    int i = start + threadIdx.x;
    int v = i < num_blocks ? block_sums[i] : 0;
    int total;
    int ex = block_exclusive_scan(v, total);
    if (i < num_blocks) block_sums[i] = ex + carry;
    __syncthreads();
    if (threadIdx.x == 0) carry += total;
    __syncthreads();
  }
}

// 第 3 步：每个元素加上所在块的起始偏移
__global__ void add_offsets(int* __restrict__ out, const int* __restrict__ block_sums, int n) {
  const int base = blockIdx.x * kTile + threadIdx.x * kItems;
  const int offset = block_sums[blockIdx.x];
#pragma unroll
  for (int k = 0; k < kItems; ++k)
    if (base + k < n) out[base + k] += offset;
}

int main() {
  const int n = 10'000'000 + 7;
  std::vector<int> h(n), ref(n), got(n);
  std::mt19937 gen(1);
  std::uniform_int_distribution<int> dist(0, 9);
  for (auto& v : h) v = dist(gen);
  int acc = 0;
  for (int i = 0; i < n; ++i) { ref[i] = acc; acc += h[i]; }

  const int num_blocks = (n + kTile - 1) / kTile;
  int *d_in, *d_out, *d_sums;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sums, num_blocks * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  auto run = [&] {
    scan_tiles<<<num_blocks, kThreads>>>(d_in, d_out, d_sums, n);
    scan_block_sums<<<1, kThreads>>>(d_sums, num_blocks);
    add_offsets<<<num_blocks, kThreads>>>(d_out, d_sums, n);
  };
  run();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (int i = 0; i < n; ++i) bad += got[i] != ref[i];
  std::printf("exclusive scan of %d ints: %s (%zu mismatches)\n", n, bad ? "FAIL" : "PASS", bad);

  float ms = time_ms(run);
  // 读 in、写 out、再读写一次 out（第 3 步）
  std::printf("time %.3f ms, %.1f GB/s effective (4 passes over the data)\n", ms, gbps(4.0 * n * sizeof(int), ms));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_sums));
  return bad ? 1 : 0;
}
