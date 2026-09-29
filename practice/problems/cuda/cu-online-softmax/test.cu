#include "judge.cuh"
#include "user.cu"

static void run_case(const char* name, int rows, int cols, float scale, bool perf = false) {
  auto h = pj::randn((size_t)rows * cols, rows * 7 + cols);
  for (auto& v : h) v *= scale;
  float* x = pj::to_device(h);
  float* y = pj::device_empty<float>((size_t)rows * cols);
  launch_softmax(x, y, rows, cols);
  if (pj::launch_ok(name)) {
    std::vector<float> want((size_t)rows * cols);
    for (int r = 0; r < rows; ++r) {
      double m = -1e300, s = 0;
      for (int c = 0; c < cols; ++c) m = std::max(m, (double)h[(size_t)r * cols + c]);
      for (int c = 0; c < cols; ++c) s += std::exp(h[(size_t)r * cols + c] - m);
      for (int c = 0; c < cols; ++c) want[(size_t)r * cols + c] = std::exp(h[(size_t)r * cols + c] - m) / s;
    }
    if (pj::check_close(name, pj::to_host(y, (size_t)rows * cols), want, 1e-3f, 1e-6f) && perf) {
      float ms = pj::timeit([&] { launch_softmax(x, y, rows, cols); });
      pj::perf(name, ms, 2.0 * rows * cols * sizeof(float));
    }
  }
  cudaFree(x);
  cudaFree(y);
}

int main() {
  run_case("small", 3, 50, 1.f);
  run_case("cols_513", 2, 513, 1.f);
  run_case("large_values", 2, 300, 1000.f);
  run_case("perf", pj::kEmulator ? 4 : 4096, pj::kEmulator ? 1000 : 8192, 1.f, true);
  return 0;
}
