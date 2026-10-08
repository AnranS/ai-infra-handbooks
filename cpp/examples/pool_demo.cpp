#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "thread_pool.hpp"

// 把 [0, n) 切成若干段交给线程池，等所有段完成
template <class F>
void parallel_for(ThreadPool& pool, int n, int chunks, F f) {
  std::vector<std::future<void>> fs;
  for (int c = 0; c < chunks; ++c) {
    int lo = n * c / chunks, hi = n * (c + 1) / chunks;
    fs.push_back(pool.submit([=] {
      for (int i = lo; i < hi; ++i) f(i);
    }));
  }
  for (auto& fu : fs) fu.get();   // 任何一段抛出的异常会在这里重新抛出
}

int main() {
  ThreadPool pool(4);

  std::vector<std::future<long>> results;
  for (long i = 0; i < 100; ++i) results.push_back(pool.submit([i] { return i * i; }));
  long sum = 0;
  for (auto& r : results) sum += r.get();
  std::printf("平方和 = %ld\n", sum);

  auto bad = pool.submit([]() -> int { throw std::runtime_error("分词失败：非法的 UTF-8"); });
  try {
    bad.get();
  } catch (const std::exception& e) {
    std::printf("任务里的异常在 get() 时抛出：%s\n", e.what());
  }

  std::vector<int> shard_tokens(8);   // 每个分片由一个任务写自己的那一格
  parallel_for(pool, 8, 3, [&](int i) { shard_tokens[i] = (i + 1) * 1000; });
  std::printf("8 个分片共 %d 个 token\n", std::accumulate(shard_tokens.begin(), shard_tokens.end(), 0));
}
