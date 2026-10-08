// g++ -std=c++20 -O2 bench_containers.cpp && ./a.out
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <list>
#include <map>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

template <class F>
double ms(F&& f, int reps = 5) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return best;
}

volatile long long sink;   // 防止编译器把结果没人用的计算整个删掉

int main() {
  const int n = 10'000'000;
  std::vector<int> v(n, 1);
  std::list<int> l(v.begin(), v.end());
  std::printf("遍历求和 %d 个 int：vector %.1f ms，list %.1f ms\n", n,
              ms([&] { sink = std::accumulate(v.begin(), v.end(), 0LL); }),
              ms([&] { sink = std::accumulate(l.begin(), l.end(), 0LL); }));

  const int keys = 4096, queries = 1'000'000;
  std::mt19937 rng(0);
  std::vector<int> ks(keys);
  for (int i = 0; i < keys; ++i) ks[i] = i * 3;
  std::vector<int> qs(queries);
  for (auto& q : qs) q = ks[rng() % keys];
  std::map<int, int> m;
  std::unordered_map<int, int> um;
  std::vector<std::pair<int, int>> sorted;
  for (int k : ks) {
    m[k] = k;
    um[k] = k;
    sorted.push_back({k, k});
  }
  std::printf("%d 个键里查 %d 次：map %.1f ms，unordered_map %.1f ms，有序 vector + 二分 %.1f ms\n", keys, queries,
              ms([&] { long long s = 0; for (int q : qs) s += m.find(q)->second; sink = s; }),
              ms([&] { long long s = 0; for (int q : qs) s += um.find(q)->second; sink = s; }),
              ms([&] {
                long long s = 0;
                for (int q : qs) s += std::lower_bound(sorted.begin(), sorted.end(), std::pair{q, 0})->second;
                sink = s;
              }));
}
