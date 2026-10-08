#include <atomic>
#include <cstdio>
#include <thread>

struct Config {
  int max_batch;
  int page_size;
};
Config config;
std::atomic<bool> ready{false};

int main() {
  std::thread loader([] {
    config = {256, 16};
    ready.store(true, std::memory_order_release);    // 之前的写（config）都"发布"出去
  });
  std::thread worker([] {
    while (!ready.load(std::memory_order_acquire)) { // 读到 true 之后，config 一定是新的
    }
    std::printf("max_batch=%d page_size=%d\n", config.max_batch, config.page_size);
  });
  loader.join();
  worker.join();
}
