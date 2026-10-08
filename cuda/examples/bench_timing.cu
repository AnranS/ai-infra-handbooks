// bench_timing.cu —— 这一节的三个实验：口径差异、工作集与缓存、短 kernel 的启动延迟
// 编译：nvcc -O3 -arch=sm_75 bench_timing.cu -o bench_timing
// 在自己的卡上建议改成 -arch=native，省掉第一次运行的 JIT
#include "common.cuh"

__global__ void vector_add(const float* a, const float* b, float* c, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void copy_kernel(const float* src, float* dst, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

__global__ void empty_kernel() {}

static void report_env() {
  int dev = 0, rt = 0, drv = 0, clk_khz = 0, mem_khz = 0, bus = 0, l2 = 0;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
  double peak = 2.0 * mem_khz * 1e3 * bus / 8.0 / 1e9;   // DDR 类显存每个时钟传两次
  std::printf("GPU: %s (sm_%d%d, %d SM, L2 %.1f MiB)\n", p.name, p.major, p.minor, p.multiProcessorCount,
              l2 / 1048576.0);
  std::printf("SM 时钟 %.0f MHz, 显存 %.0f MHz x %d bit -> 理论峰值带宽 %.0f GB/s\n",
              clk_khz / 1000.0, mem_khz / 1000.0, bus, peak);
  std::printf("CUDA runtime %d.%d, driver %d.%d\n\n", rt / 1000, rt % 1000 / 10, drv / 1000, drv % 1000 / 10);
}

static double peak_gbps() {
  int dev = 0, mem_khz = 0, bus = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));
  return 2.0 * mem_khz * 1e3 * bus / 8.0 / 1e9;
}

// 实验一：同一个 kernel，三种统计口径给出三个数
static void experiment_scope() {
  const long long n = 4096LL * 4096;                      // 1600 万个元素，三个数组共 192 MiB
  const size_t bytes = n * sizeof(float);
  float *a, *b, *c;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMalloc(&c, bytes));
  std::vector<float> ha(n, 1.f), hb(n, 2.f), hc(n);
  CUDA_CHECK(cudaMemcpy(a, ha.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b, hb.data(), bytes, cudaMemcpyHostToDevice));
  int block = 256;
  long long grid = (n + block - 1) / block;
  auto run = [&] { vector_add<<<(unsigned)grid, block>>>(a, b, c, n); };

  float batch = time_ms(run, 200, 20);                    // 两个 event 包住 200 次，总时间 / 200
  auto [lo, med] = time_stats(run, 200, 20);              // 逐次打点的 200 个样本
  double moved = 3.0 * n * sizeof(float);                 // 算法字节数：读两个数组、写一个
  std::printf("--- 实验一：同一个向量加法，三种口径 ---\n");
  std::printf("%-18s %10s %12s\n", "统计量", "毫秒", "GB/s");
  std::printf("%-18s %10.6f %12.1f\n", "整批平均", batch, gbps(moved, batch));
  std::printf("%-18s %10.6f %12.1f\n", "逐次中位数", med, gbps(moved, med));
  std::printf("%-18s %10.6f %12.1f\n", "逐次最小值", lo, gbps(moved, lo));

  CUDA_CHECK(cudaMemcpy(hc.data(), c, bytes, cudaMemcpyDeviceToHost));
  std::vector<float> ref(n, 3.f);                         // 校验在计时区间之外
  std::printf("校验：");
  check_close(hc.data(), ref.data(), n);
  std::printf("\n");
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
}

// 实验二：复制不同大小的数组，看"有效带宽"怎样随工作集变化
static void experiment_working_set() {
  const double peak = peak_gbps();
  std::printf("--- 实验二：工作集大小与有效带宽（复制，一读一写）---\n");
  std::printf("%10s %12s %10s %10s %8s %s\n", "单数组 MiB", "读写 MiB", "中位 ms", "GB/s", "占峰值", "校验");
  for (int mib : {1, 4, 16, 64, 256}) {
    long long n = (long long)mib * 1048576 / sizeof(float);
    size_t bytes = n * sizeof(float);
    float *src, *dst;
    CUDA_CHECK(cudaMalloc(&src, bytes));
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    std::vector<float> h(n);
    fill_random(h, 7);
    CUDA_CHECK(cudaMemcpy(src, h.data(), bytes, cudaMemcpyHostToDevice));
    int block = 256;
    long long grid = (n + block - 1) / block;
    auto run = [&] { copy_kernel<<<(unsigned)grid, block>>>(src, dst, n); };
    auto [lo, med] = time_stats(run, 200, 20);
    (void)lo;
    double moved = 2.0 * bytes;
    std::vector<float> back(n);
    CUDA_CHECK(cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) ok = back[i] == h[i];
    std::printf("%10d %12.0f %10.6f %10.1f %7.0f%% %s\n", mib, 2.0 * mib, med, gbps(moved, med),
                100.0 * gbps(moved, med) / peak, ok ? "PASS" : "FAIL");
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
  }
  std::printf("\n");
}

// 实验三：kernel 越短，测量值里混进的启动延迟占比越大
static void experiment_overhead() {
  auto run = [] { empty_kernel<<<1, 1>>>(); };
  float batch = time_ms(run, 1000, 50);
  auto [lo, med] = time_stats(run, 1000, 50);
  std::printf("--- 实验三：一个什么都不做的 kernel ---\n");
  std::printf("整批平均 %.6f ms，逐次中位数 %.6f ms，逐次最小值 %.6f ms\n", batch, med, lo);
  std::printf("逐次测量的这个下限就是「排队 + 提交」的开销；kernel 自身耗时接近它时，event 的读数基本在量开销\n\n");
}

int main() {
  report_env();
  experiment_scope();
  experiment_working_set();
  experiment_overhead();
  return 0;
}
