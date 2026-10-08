#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
  std::atomic<bool> weights_loaded{false};
  std::atomic<int> started{0};
  std::vector<std::thread> workers;
  for (int i = 0; i < 3; ++i) {
    workers.emplace_back([&] {
      weights_loaded.wait(false, std::memory_order_acquire);   // 值还是 false 就睡眠
      started.fetch_add(1, std::memory_order_relaxed);
    });
  }
  weights_loaded.store(true, std::memory_order_release);        // 加载完成
  weights_loaded.notify_all();
  for (auto& w : workers) w.join();
  std::printf("%d 个工作线程都开始了\n", started.load());
}
