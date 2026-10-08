// device_query.cu —— 打印 GPU 的关键参数
// 编译：nvcc -O3 -arch=sm_75 device_query.cu -o device_query
#include "common.cuh"

int main() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int dev = 0; dev < count; ++dev) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int mem_clock_khz = 0, bus_width = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, dev));
    // DDR 类显存每个时钟传两次数据
    double peak_bw = 2.0 * mem_clock_khz * 1e3 * (bus_width / 8) / 1e9;

    std::printf("GPU %d: %s (sm_%d%d)\n", dev, p.name, p.major, p.minor);
    std::printf("  SMs                         : %d\n", p.multiProcessorCount);
    std::printf("  global memory               : %.1f GB\n", p.totalGlobalMem / 1e9);
    std::printf("  theoretical bandwidth       : %.0f GB/s\n", peak_bw);
    std::printf("  L2 cache                    : %d KB\n", p.l2CacheSize / 1024);
    std::printf("  shared memory per SM        : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    std::printf("  shared memory per block     : %zu KB (opt-in max %zu KB)\n",
                p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
    std::printf("  registers per SM / block    : %d / %d\n", p.regsPerMultiprocessor, p.regsPerBlock);
    std::printf("  max threads per SM / block  : %d / %d\n", p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock);
    std::printf("  max blocks per SM           : %d\n", p.maxBlocksPerMultiProcessor);
    std::printf("  warp size                   : %d\n", p.warpSize);
  }
  return 0;
}
