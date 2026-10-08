// nccl_allreduce.cu —— 单进程多 GPU 的 NCCL all-reduce
// 编译：nvcc -O3 -arch=sm_75 nccl_allreduce.cu -o nccl_allreduce -lnccl
#include "common.cuh"
#include <nccl.h>

#define NCCL_CHECK(call)                                                                 \
  do {                                                                                   \
    ncclResult_t r_ = (call);                                                            \
    if (r_ != ncclSuccess) {                                                             \
      std::fprintf(stderr, "NCCL error %s at %s:%d\n", ncclGetErrorString(r_), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                           \
    }                                                                                    \
  } while (0)

__global__ void fill(float* x, int n, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = value + i % 7;
}

int main() {
  int ngpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ngpus));
  if (ngpus < 2) {
    std::printf("SKIP: needs at least 2 GPUs, found %d\n", ngpus);
    return 0;
  }
  const int n = 32 * 1024 * 1024;   // 每张卡 128 MB
  std::vector<int> devs(ngpus);
  for (int i = 0; i < ngpus; ++i) devs[i] = i;
  std::vector<ncclComm_t> comms(ngpus);
  NCCL_CHECK(ncclCommInitAll(comms.data(), ngpus, devs.data()));

  std::vector<float*> buf(ngpus);
  std::vector<cudaStream_t> streams(ngpus);
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMalloc(&buf[g], n * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&streams[g]));
    fill<<<(n + 255) / 256, 256, 0, streams[g]>>>(buf[g], n, static_cast<float>(g));   // 第 g 张卡：g + i % 7
    CUDA_CHECK_LAST();
  }

  auto allreduce = [&] {
    // 一个线程为多个通信器发起集合通信时，必须用 group 包起来，否则会死锁
    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < ngpus; ++g)
      NCCL_CHECK(ncclAllReduce(buf[g], buf[g], n, ncclFloat, ncclSum, comms[g], streams[g]));   // 原地
    NCCL_CHECK(ncclGroupEnd());
  };
  allreduce();
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(streams[g]));
  }

  // 期望：sum_g (g + i % 7) = ngpus * (ngpus - 1) / 2 + ngpus * (i % 7)
  std::vector<float> h(n);
  size_t bad = 0;
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMemcpy(h.data(), buf[g], n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) bad += h[i] != ngpus * (ngpus - 1) / 2.f + ngpus * static_cast<float>(i % 7);
  }
  std::printf("all-reduce across %d GPUs: %s (%zu mismatches)\n", ngpus, bad ? "FAIL" : "PASS", bad);

  // 计时：以卡 0 的流为准（其他卡同步完成）
  CUDA_CHECK(cudaSetDevice(0));
  GpuTimer t;
  const int iters = 20;
  t.start(streams[0]);
  for (int it = 0; it < iters; ++it) allreduce();
  float ms = t.stop(streams[0]) / iters;
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(streams[g]));
  }
  const double algbw = n * sizeof(float) / (ms * 1e-3) / 1e9;
  std::printf("%.3f ms, algbw %.1f GB/s, busbw %.1f GB/s\n", ms, algbw, algbw * 2.0 * (ngpus - 1) / ngpus);

  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaFree(buf[g]));
    CUDA_CHECK(cudaStreamDestroy(streams[g]));
    NCCL_CHECK(ncclCommDestroy(comms[g]));
  }
  return bad ? 1 : 0;
}
