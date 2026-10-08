#include <cstdio>
#include <thread>
#include <vector>

#include "blocking_queue.hpp"

struct Req {
  int id;
  int tokens;
};

int main() {
  BlockingQueue<Req> q;
  std::thread producer([&] {
    for (int i = 0; i < 100; ++i) q.push({i, i % 10 + 1});
    q.close();                     // 生产完毕
  });
  std::vector<long> handled(2, 0), tokens(2, 0);   // 每个消费者只写自己的那一格：没有共享
  std::vector<std::thread> consumers;
  for (int c = 0; c < 2; ++c) {
    consumers.emplace_back([&, c] {
      while (auto r = q.pop()) {
        ++handled[c];
        tokens[c] += r->tokens;
      }
    });
  }
  producer.join();
  for (auto& t : consumers) t.join();
  std::printf("处理了 %ld 个请求，共 %ld 个 token\n", handled[0] + handled[1], tokens[0] + tokens[1]);
}
