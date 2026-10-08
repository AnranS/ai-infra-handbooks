// g++ -std=c++20 -O2 traverse.cpp && ./a.out
#include <chrono>
#include <cstdio>
#include <vector>

volatile float sink;

int main() {
  const int n = 4096;
  std::vector<float> m(std::size_t(n) * n, 1.0f);   // 行主序：m[i * n + j]
  auto time = [&](bool by_row) {
    auto t0 = std::chrono::steady_clock::now();
    float s = 0;
    for (int a = 0; a < n; ++a)
      for (int b = 0; b < n; ++b) s += by_row ? m[std::size_t(a) * n + b] : m[std::size_t(b) * n + a];
    sink = s;
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  time(true);   // 预热
  std::printf("按行遍历 %.0f ms，按列遍历 %.0f ms\n", time(true), time(false));
}
