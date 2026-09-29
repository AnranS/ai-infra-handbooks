#include <thread>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

int main() {
  pj::run("example_capacity", [] {
    SpscQueue<int, 4> q;
    for (int i = 0; i < 4; ++i) pj::require(q.try_push(i), "容量内的 push 应该成功");
    pj::require(!q.try_push(99), "满了之后 try_push 返回 false");
    int v = -1;
    pj::require(q.try_pop(v) && v == 0, "先进先出");
    pj::require(q.try_push(4), "取出一个之后又能放");
  });
  pj::run("wraparound", [] {
    SpscQueue<int, 8> q;
    int next_in = 0, next_out = 0, v;
    for (int round = 0; round < 1000; ++round) {
      for (int i = 0; i < 5; ++i) pj::require(q.try_push(next_in++), "push");
      for (int i = 0; i < 5; ++i) {
        pj::require(q.try_pop(v), "pop");
        pj::require_eq(v, next_out++, "绕回之后的顺序");
      }
    }
    pj::require(!q.try_pop(v), "取空后 try_pop 返回 false");
  });
  pj::run("concurrent_order", [] {
    SpscQueue<long, 256> q;
    const long n = 100000;
    std::thread producer([&] {
      for (long i = 1; i <= n; ++i)
        while (!q.try_push(i)) std::this_thread::yield();
    });
    long expect = 1, v;
    while (expect <= n) {
      if (!q.try_pop(v)) {
        std::this_thread::yield();
        continue;
      }
      if (v != expect) break;
      ++expect;
    }
    producer.join();
    pj::require_eq(expect, n + 1, "按顺序收到的元素个数（加一）");
  });
}
