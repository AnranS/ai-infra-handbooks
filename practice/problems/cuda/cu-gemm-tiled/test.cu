#include "judge.cuh"
#include "user.cu"

static void run_case(const char* name, int M, int N, int K, bool perf = false) {
  auto hA = pj::randn((size_t)M * K, 1), hB = pj::randn((size_t)K * N, 2);
  float* A = pj::to_device(hA);
  float* B = pj::to_device(hB);
  float* C = pj::device_empty<float>((size_t)M * N);
  launch_gemm(A, B, C, M, N, K);
  if (pj::launch_ok(name)) {
    auto got = pj::to_host(C, (size_t)M * N);
    // 抽查：大矩阵只检查一部分元素
    int step = (long long)M * N > 100000 ? 97 : 1;
    std::vector<float> g, w;
    for (long long idx = 0; idx < (long long)M * N; idx += step) {
      int i = idx / N, j = idx % N;
      double s = 0;
      for (int k = 0; k < K; ++k) s += (double)hA[(size_t)i * K + k] * hB[(size_t)k * N + j];
      g.push_back(got[idx]);
      w.push_back((float)s);
    }
    if (pj::check_close(name, g, w, 1e-3f, 1e-3f) && perf) {
      float ms = pj::timeit([&] { launch_gemm(A, B, C, M, N, K); }, 10);
      pj::perf(name, ms, 4.0 * ((double)M * K + (double)K * N + (double)M * N), 2.0 * M * N * K);
    }
  }
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}

int main() {
  run_case("small_16", 16, 16, 16);
  run_case("odd_37x29x53", 37, 29, 53);
  int n = pj::kEmulator ? 48 : 2048;
  run_case("perf", n, n, n, true);
  return 0;
}
