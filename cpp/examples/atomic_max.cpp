#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

void atomic_max(std::atomic<long>& target, long v) {
  long cur = target.load(std::memory_order_relaxed);
  while (cur < v && !target.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    // 失败时 cur 已经被更新成当前值，循环回去重新比较
  }
}

int main() {
  std::atomic<long> peak{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t) {
    ts.emplace_back([&, t] {
      for (long i = 0; i < 10000; ++i) atomic_max(peak, (i * 7 + t * 13) % 9973);
    });
  }
  for (auto& th : ts) th.join();
  std::printf("峰值 = %ld\n", peak.load());
}
