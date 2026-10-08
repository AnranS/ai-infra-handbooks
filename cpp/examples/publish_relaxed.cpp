#include <atomic>
#include <cstdio>
#include <thread>

struct Config {
  int max_batch;
  int page_size;
};
Config config;                    // 普通变量
std::atomic<bool> ready{false};

int main() {
  std::thread loader([] {
    config = {256, 16};
    ready.store(true, std::memory_order_relaxed);   // 只保证原子，不保证 config 先写完
  });
  std::thread worker([] {
    while (!ready.load(std::memory_order_relaxed)) {
    }
    std::printf("%d\n", config.max_batch);           // 可能读到旧值：数据竞争
  });
  loader.join();
  worker.join();
}
