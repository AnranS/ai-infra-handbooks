// g++ -std=c++20 -O2 aos_soa.cpp && ./a.out
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

struct RequestAoS {   // 一个请求的所有字段放在一起：64 字节
  std::int64_t id;
  std::int32_t prompt_len, output_len;
  float temperature, top_p;
  std::int32_t num_blocks, priority;
  bool finished, streaming;
  char padding[30];
};

struct RequestsSoA {  // 每个字段一个数组
  std::vector<std::int64_t> id;
  std::vector<std::int32_t> prompt_len, output_len;
};

template <class F>
double best_ms(F&& f) {
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  return best;
}

volatile std::int64_t sink;

int main() {
  const int n = 4'000'000;
  std::vector<RequestAoS> aos(n);
  RequestsSoA soa;
  soa.output_len.resize(n);
  for (int i = 0; i < n; ++i) aos[i].output_len = soa.output_len[i] = i % 100;
  // 调度器每一步只关心一个字段：统计所有请求已经生成的 token 数
  double t_aos = best_ms([&] { std::int64_t s = 0; for (const auto& r : aos) s += r.output_len; sink = s; });
  double t_soa = best_ms([&] { std::int64_t s = 0; for (int v : soa.output_len) s += v; sink = s; });
  std::printf("sizeof(RequestAoS)=%zu\n", sizeof(RequestAoS));
  std::printf("只读一个字段：AoS %.2f ms，SoA %.2f ms\n", t_aos, t_soa);
}
