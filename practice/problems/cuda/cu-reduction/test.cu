#include "judge.cuh"
#include "user.cu"

static void run_case(const char* name, int n, bool perf = false) {
  auto h = pj::randn(n, n);
  double want = 0;
  for (float v : h) want += v;
  float* d = pj::to_device(h);
  float got = reduce_sum(d, n);
  if (pj::launch_ok(name)) {
    if (std::fabs(got - want) <= 1e-3 + 1e-4 * std::fabs(want) + 1e-6 * n) {
      pj::pass(name);
      if (perf) {
        float ms = pj::timeit([&] { reduce_sum(d, n); }, 10);
        pj::perf(name, ms, 1.0 * n * sizeof(float));
      }
    } else {
      pj::fail(name, "期望 " + std::to_string(want) + "，实际 " + std::to_string(got));
    }
  }
  cudaFree(d);
}

int main() {
  run_case("n_1", 1);
  run_case("n_513", 513);
  run_case("n_100003", pj::kEmulator ? 3001 : 100003);
  run_case("perf", pj::kEmulator ? 5000 : (1 << 26), true);
  return 0;
}
