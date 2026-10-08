// common.cuh —— 本手册所有示例共用的小工具
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

// 检查 CUDA API 的返回值，出错时打印位置并退出
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                     \
                   cudaGetErrorName(err_), __FILE__, __LINE__,                 \
                   cudaGetErrorString(err_));                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// kernel 启动之后调用：立即报告启动配置错误
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// 用 CUDA event 计时，单位毫秒
struct GpuTimer {
  cudaEvent_t start_, stop_;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }
  float stop(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(stop_, s));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

// 先预热 warmup 次，再运行 iters 次取平均耗时（毫秒）
template <typename F>
float time_ms(F&& fn, int iters = 20, int warmup = 3) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaGetLastError());
  GpuTimer t;
  t.start();
  for (int i = 0; i < iters; ++i) fn();
  float ms = t.stop() / iters;
  CUDA_CHECK(cudaGetLastError());
  return ms;
}

// 逐次采样：返回（最小值，中位数），单位毫秒。和 time_ms 的"整批平均"是两种口径，不能混着比
template <typename F>
std::pair<float, float> time_stats(F&& fn, int iters = 50, int warmup = 5) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> ms(iters);
  GpuTimer t;
  for (int i = 0; i < iters; ++i) {
    t.start();
    fn();
    ms[i] = t.stop();                               // 每次都同步一下，拿到的是单次样本
  }
  CUDA_CHECK(cudaGetLastError());
  std::sort(ms.begin(), ms.end());
  return {ms.front(), ms[iters / 2]};
}

inline void fill_random(std::vector<float>& v, unsigned seed = 42, float lo = -1.f, float hi = 1.f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& x : v) x = dist(gen);
}

// 与 CPU 参考结果比较：|got - ref| <= atol + rtol * |ref| 视为正确
inline bool check_close(const float* got, const float* ref, size_t n,
                        float rtol = 1e-4f, float atol = 1e-5f) {
  size_t bad = 0;
  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
    max_err = std::max(max_err, err);
    if (!(err <= atol + rtol * std::fabs(ref[i]))) {  // 取反写法能同时捕获 NaN
      if (bad < 5) std::fprintf(stderr, "  mismatch at %zu: got %g, expected %g\n", i, got[i], ref[i]);
      ++bad;
    }
  }
  std::printf("%s  max_abs_err=%.3e  mismatches=%zu/%zu\n", bad ? "FAIL" : "PASS", max_err, bad, n);
  return bad == 0;
}

// 当前 GPU 的计算能力低于要求时打印 SKIP 并正常退出，方便批量运行示例
inline void require_sm(int major, int minor = 0) {
  int dev = 0;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  if (p.major * 10 + p.minor < major * 10 + minor) {
    std::printf("SKIP: needs sm_%d%d, this GPU (%s) is sm_%d%d\n", major, minor, p.name, p.major, p.minor);
    std::exit(0);
  }
}

inline int sm_count() {
  int dev = 0, n = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  return n;
}

// 带宽（GB/s）与算力（TFLOPS）换算
inline double gbps(double bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }
inline double tflops(double flops, float ms) { return flops / (ms * 1e-3) / 1e12; }
