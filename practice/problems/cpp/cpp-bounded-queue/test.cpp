#include <atomic>
#include <chrono>
#include <optional>
#include <thread>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

using namespace std::chrono_literals;

int main() {
  pj::run("example_fifo", [] {
    BoundedQueue<int> q(4);
    for (int i = 0; i < 3; ++i) q.push(i);
    pj::require_eq(q.size(), std::size_t(3), "size");
    for (int i = 0; i < 3; ++i) pj::require_eq(*q.pop(), i, "先进先出");
  });
  pj::run("close_semantics", [] {
    BoundedQueue<int> q(4);
    q.push(1);
    q.push(2);
    q.close();
    pj::require(!q.push(3), "关闭后 push 返回 false");
    pj::require_eq(*q.pop(), 1, "关闭前的元素仍然能取出");
    pj::require_eq(*q.pop(), 2, "关闭前的元素仍然能取出");
    pj::require(!q.pop().has_value(), "取空后返回 nullopt");
  });
  pj::run("blocks_when_full", [] {
    BoundedQueue<int> q(2);
    q.push(1);
    q.push(2);
    std::atomic<bool> pushed{false};
    std::thread producer([&] {
      q.push(3);   // 满了：应该阻塞
      pushed = true;
    });
    std::this_thread::sleep_for(100ms);
    pj::require(!pushed.load(), "队列满时 push 应该阻塞");
    pj::require_eq(*q.pop(), 1, "pop");
    producer.join();
    pj::require(pushed.load(), "空出位置后 push 应该完成");
    pj::require_eq(q.size(), std::size_t(2), "size");
  });
  pj::run("blocked_consumer_wakes_on_close", [] {
    BoundedQueue<int> q(2);
    std::optional<int> got = 42;
    std::thread consumer([&] { got = q.pop(); });
    std::this_thread::sleep_for(50ms);
    q.close();
    consumer.join();
    pj::require(!got.has_value(), "等待中的消费者在关闭后应该醒来并得到 nullopt");
  });
  pj::run("mpmc", [] {
    BoundedQueue<long> q(8);
    const int producers = 4, per = 2000;
    std::vector<std::thread> ps, cs;
    std::atomic<long> sum{0}, count{0};
    for (int p = 0; p < producers; ++p)
      ps.emplace_back([&, p] {
        for (int i = 1; i <= per; ++i) q.push(long(p) * per + i);
      });
    for (int c = 0; c < 3; ++c)
      cs.emplace_back([&] {
        while (auto v = q.pop()) {
          sum += *v;
          ++count;
        }
      });
    for (auto& t : ps) t.join();
    q.close();
    for (auto& t : cs) t.join();
    long n = long(producers) * per;
    pj::require_eq(count.load(), n, "每个元素恰好取出一次");
    pj::require_eq(sum.load(), n * (n + 1) / 2, "元素之和");
  });
}
