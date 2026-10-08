#include <atomic>
#include <chrono>
#include <cstdio>
#include <stop_token>
#include <thread>

int main() {
  std::atomic<int> ticks{0};
  {
    std::jthread reporter([&](std::stop_token st) {
      while (!st.stop_requested()) {
        ticks.fetch_add(1);        // 假装在上报一次指标
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      std::printf("reporter 收到停止请求，退出\n");
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }   // jthread 析构：先 request_stop()，再 join()
  std::printf("上报过：%s\n", ticks.load() > 0 ? "是" : "否");
}
