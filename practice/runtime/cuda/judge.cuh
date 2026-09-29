// CUDA C++ 练习题的测试工具：在真 GPU（nvcc）和 CPU 模拟器（g++，定义了 PRACTICE_EMU）上都能编译。
// 测试程序按行输出结果，由 practice/cudajudge.py 解析：
//   CASE <名字> PASS | FAIL <说明> | ERROR <说明>
//   PERF <名字> <毫秒> <GB/s> <TFLOPS>
//   TIER <名字> <达到参照的比例> <铜> <银> <金> <参照说明>      （只在真 GPU 上输出）
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>   // 模拟器模式下是 cuda/tools/emu/include 里的同名头文件

#define CK(call)                                                                                  \
  do {                                                                                            \
    cudaError_t err_ = (call);                                                                    \
    if (err_ != cudaSuccess) {                                                                    \
      std::printf("CASE setup ERROR %s 失败：%s\n", #call, cudaGetErrorString(err_));              \
      std::exit(1);                                                                               \
    }                                                                                             \
  } while (0)

namespace pj {

#ifdef PRACTICE_EMU
constexpr bool kEmulator = true;
#else
constexpr bool kEmulator = false;
#endif

inline std::vector<float> randn(size_t n, unsigned seed = 0) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

template <class T>
T* to_device(const std::vector<T>& h) {
  T* d = nullptr;
  CK(cudaMalloc((void**)&d, h.size() * sizeof(T) + 16));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

// 像 cudaMalloc 一样的"脏"内存：填满 0xFF（对 float 是 NaN），没写到的位置会被测出来
template <class T>
T* device_empty(size_t n) {
  T* d = nullptr;
  CK(cudaMalloc((void**)&d, n * sizeof(T) + 16));
  CK(cudaMemset(d, 0xFF, n * sizeof(T)));
  return d;
}

template <class T>
std::vector<T> to_host(const T* d, size_t n) {
  std::vector<T> h(n);
  CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

// kernel 启动后检查错误（launch 配置非法、越界导致的非法地址等）
inline bool launch_ok(const char* name) {
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("CASE %s ERROR kernel 执行出错：%s\n", name, cudaGetErrorString(e));
    return false;
  }
  return true;
}

inline bool check_close(const char* name, const std::vector<float>& got, const std::vector<float>& want,
                        float rtol = 1e-5f, float atol = 1e-5f) {
  if (got.size() != want.size()) {
    std::printf("CASE %s FAIL 长度不对：期望 %zu，实际 %zu\n", name, want.size(), got.size());
    return false;
  }
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    float g = got[i], w = want[i];
    if (!(std::fabs(g - w) <= atol + rtol * std::fabs(w))) {
      if (bad++ == 0) first = i;
    }
  }
  if (bad) {
    std::printf("CASE %s FAIL %zu 个元素不对，第一个在下标 %zu：期望 %.6g，实际 %.6g\n", name, bad, first, want[first],
                got[first]);
    return false;
  }
  std::printf("CASE %s PASS\n", name);
  return true;
}

inline void pass(const char* name) { std::printf("CASE %s PASS\n", name); }
inline void fail(const char* name, const std::string& why) { std::printf("CASE %s FAIL %s\n", name, why.c_str()); }

// 计时：跑 iters 次取平均（毫秒）。模拟器上不计时。
template <class F>
float timeit(F f, int iters = 20) {
  if (kEmulator) return 0.f;
  f();
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) f();
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms / iters;
}

inline void perf(const char* name, float ms, double bytes, double flops = 0) {
  if (kEmulator) return;
  std::printf("PERF %s %.4f %.2f %.3f\n", name, ms, ms > 0 ? bytes / ms / 1e6 : 0.0, ms > 0 ? flops / ms / 1e9 : 0.0);
}

// 本机实测的显存带宽（GB/s）：256 MiB 的 device-to-device 拷贝，读 + 写都算。性能档位用它做分母，而不是规格表上的峰值。
inline double measured_bandwidth_gbs() {
  if (kEmulator) return 0;
  static double cached = 0;
  if (cached > 0) return cached;
  size_t n = size_t(256) << 20;
  void *a = nullptr, *b = nullptr;
  CK(cudaMalloc(&a, n));
  CK(cudaMalloc(&b, n));
  CK(cudaMemset(a, 0, n));
  float ms = timeit([&] { cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice); }, 20);
  cudaFree(a);
  cudaFree(b);
  cached = 2.0 * n / ms / 1e6;
  return cached;
}

// 性能档位：ratio 是达到参照（实测带宽、cuBLAS 等）的比例，bronze / silver / gold 是三档的门槛。
inline void tier(const char* name, double ratio, double bronze, double silver, double gold, const char* what) {
  if (kEmulator) return;
  std::printf("TIER %s %.4f %.2f %.2f %.2f %s\n", name, ratio, bronze, silver, gold, what);
}

// 带宽型 kernel：按实测带宽定档（铜 50%、银 80%、金 90%）
inline void bandwidth_tier(const char* name, float ms, double bytes) {
  if (kEmulator || ms <= 0) return;
  tier(name, bytes / ms / 1e6 / measured_bandwidth_gbs(), 0.5, 0.8, 0.9, "本机实测带宽");
}

}  // namespace pj
