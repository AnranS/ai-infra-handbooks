#include "judge.cuh"
#include "user.cu"

static void run_case(const char* name, int rows, int cols, unsigned seed) {
  auto h = pj::randn((size_t)rows * cols, seed);
  float* a = pj::to_device(h);
  float* b = pj::device_empty<float>((size_t)rows * cols);
  launch_transpose(a, b, rows, cols);
  if (pj::launch_ok(name)) {
    std::vector<float> want((size_t)rows * cols);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j) want[(size_t)j * rows + i] = h[(size_t)i * cols + j];
    pj::check_close(name, pj::to_host(b, (size_t)rows * cols), want, 0, 0);
  }
  cudaFree(a);
  cudaFree(b);
}

int main() {
  run_case("square_64", 64, 64, 0);
  run_case("odd_70x45", 70, 45, 1);
  run_case("thin_3x100", 3, 100, 2);
  // 性能：8192×8192 转置，读写各 256 MB（模拟器上只跑小尺寸、不计时）
  int n = pj::kEmulator ? 96 : 8192;
  auto h = pj::randn((size_t)n * n, 3);
  float* a = pj::to_device(h);
  float* b = pj::device_empty<float>((size_t)n * n);
  launch_transpose(a, b, n, n);
  if (pj::launch_ok("perf")) {
    auto got = pj::to_host(b, (size_t)n * n);
    bool ok = true;
    for (int i = 0; i < n && ok; ++i)
      for (int j = 0; j < n; ++j)
        if (got[(size_t)j * n + i] != h[(size_t)i * n + j]) { ok = false; break; }
    if (ok) {
      float ms = pj::timeit([&] { launch_transpose(a, b, n, n); });
      pj::pass("perf");
      pj::perf("perf", ms, 2.0 * n * n * sizeof(float));
    } else {
      pj::fail("perf", "大矩阵的转置结果不对");
    }
  }
  return 0;
}
