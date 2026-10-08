#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

int main() {
  long by_mutex = 0;
  std::mutex m;
  std::atomic<long> by_atomic{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < 10000; ++i) {
        {
          std::lock_guard lk(m);   // 构造时加锁，离开作用域时解锁（RAII）
          ++by_mutex;
        }
        by_atomic.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& t : ts) t.join();
  std::printf("mutex：%ld，atomic：%ld\n", by_mutex, by_atomic.load());
}
