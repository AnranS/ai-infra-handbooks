#include "judge.cuh"
#include "user.cu"

static void run_case(const char* name, int n, bool perf = false) {
  auto ha = pj::randn(n, 1), hb = pj::randn(n, 2);
  float* a = pj::to_device(ha);
  float* b = pj::to_device(hb);
  float* c = pj::device_empty<float>(n);
  launch_vector_add(a, b, c, n);
  if (pj::launch_ok(name)) {
    std::vector<float> want(n);
    for (int i = 0; i < n; ++i) want[i] = ha[i] + hb[i];
    if (pj::check_close(name, pj::to_host(c, n), want, 0, 0) && perf) {
      float ms = pj::timeit([&] { launch_vector_add(a, b, c, n); });
      pj::perf(name, ms, 3.0 * n * sizeof(float));
    }
  }
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

int main() {
  run_case("n_1", 1);
  run_case("n_257", 257);
  run_case("n_1000", 1000);
  run_case("perf", pj::kEmulator ? 4096 : (1 << 26), true);   // 64M 个 float，每次搬运 768 MB
  return 0;
}
