// g++ -std=c++20 -O2 -pthread false_sharing.cpp && ./a.out
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

struct Plain {
  std::atomic<long> value{0};
};
struct alignas(64) Padded {   // 每个计数器独占一个 64 字节的缓存行
  std::atomic<long> value{0};
};

template <class Counter>
double run(int threads, long iters) {
  std::vector<Counter> counters(threads);
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; ++t)
    ts.emplace_back([&, t] {
      for (long i = 0; i < iters; ++i) counters[t].value.fetch_add(1, std::memory_order_relaxed);
    });
  for (auto& th : ts) th.join();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
  const int threads = 4;
  const long iters = 20'000'000;
  std::printf("sizeof(Plain)=%zu sizeof(Padded)=%zu\n", sizeof(Plain), sizeof(Padded));
  std::printf("%d 个线程各加自己的计数器 %ld 次：挨在一起 %.0f ms，按缓存行隔开 %.0f ms\n", threads, iters,
              run<Plain>(threads, iters), run<Padded>(threads, iters));
}
