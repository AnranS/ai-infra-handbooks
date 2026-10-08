#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

class Latch {
 public:
  explicit Latch(int n) : count_(n) {}
  void count_down() {
    if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) count_.notify_all();   // 最后一个到达的负责唤醒
  }
  void wait() {
    int c = count_.load(std::memory_order_acquire);
    while (c != 0) {
      count_.wait(c, std::memory_order_acquire);   // 值还是 c 就睡眠
      c = count_.load(std::memory_order_acquire);
    }
  }

 private:
  std::atomic<int> count_;
};

int main() {
  const int n = 4;
  Latch ready(n);
  int loaded[n] = {};
  std::vector<std::thread> shards;
  for (int i = 0; i < n; ++i) {
    shards.emplace_back([&, i] {
      loaded[i] = 100 + i;   // 各自加载一个权重分片
      ready.count_down();
    });
  }
  ready.wait();               // 所有分片都加载完才开始服务
  int sum = 0;
  for (int v : loaded) sum += v;
  std::printf("所有分片就绪，校验和 %d\n", sum);
  for (auto& t : shards) t.join();
}
