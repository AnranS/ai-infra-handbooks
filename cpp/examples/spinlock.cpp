#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

class SpinLock {
 public:
  void lock() {
    while (flag_.test_and_set(std::memory_order_acquire)) {   // 抢锁
      while (flag_.test(std::memory_order_relaxed)) {         // 抢不到时只读不写，避免缓存行来回失效
      }
    }
  }
  void unlock() { flag_.clear(std::memory_order_release); }

 private:
  std::atomic_flag flag_;   // C++20 起默认初始化为"未设置"
};

int main() {
  SpinLock lock;
  long counter = 0;
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < 10000; ++i) {
        std::lock_guard g(lock);   // 任何有 lock()/unlock() 的类型都能配 lock_guard
        ++counter;
      }
    });
  }
  for (auto& th : ts) th.join();
  std::printf("counter = %ld\n", counter);
}
