#include <cstdio>
#include <thread>

#include "spsc_queue.hpp"

int main() {
  SpscQueue<long, 1024> q;
  const long n = 200000;
  std::thread producer([&] {
    for (long i = 1; i <= n; ++i) {
      while (!q.try_push(i)) std::this_thread::yield();   // 满了就让出 CPU
    }
  });
  long received = 0, sum = 0, last = 0;
  bool in_order = true;
  while (received < n) {
    long v;
    if (!q.try_pop(v)) {
      std::this_thread::yield();
      continue;
    }
    in_order = in_order && v == last + 1;
    last = v;
    sum += v;
    ++received;
  }
  producer.join();
  std::printf("收到 %ld 个，%s，和 = %ld\n", received, in_order ? "顺序正确" : "顺序错乱", sum);
}
