#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

static std::string show(const std::vector<std::size_t>& v) {
  std::string s = "{";
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
  return s + "}";
}

struct S1 { bool a; double b; int c; };
struct S2 { double b; int c; bool a; };
struct S3 { char a; short b; char c; long long d; float e; };

int main() {
  pj::run("example", [] {
    Layout l = layout_of({{1, 1}, {8, 8}, {4, 4}});
    pj::require_eq(show(l.offsets), std::string("{0, 8, 16}"), "struct { bool; double; int; } 的偏移");
    pj::require_eq(l.size, std::size_t(24), "它的大小");
    pj::require_eq(show(best_order({{1, 1}, {8, 8}, {4, 4}})), std::string("{1, 2, 0}"), "最省空间的顺序");
  });
  pj::run("align_up", [] {
    constexpr std::size_t c = align_up(13, 4);  // align_up 要能在编译期求值
    pj::require(align_up(0, 8) == 0 && align_up(1, 8) == 8 && align_up(8, 8) == 8 && c == 16, "align_up 的基本情况");
    pj::require_eq(align_up(4097, 4096), std::size_t(8192), "按页对齐");
  });
  pj::run("matches_compiler", [] {
    Layout l1 = layout_of({{sizeof(bool), alignof(bool)}, {sizeof(double), alignof(double)}, {sizeof(int), alignof(int)}});
    pj::require_eq(show(l1.offsets), show({offsetof(S1, a), offsetof(S1, b), offsetof(S1, c)}), "S1 的偏移和编译器一致");
    pj::require_eq(l1.size, sizeof(S1), "S1 的大小和编译器一致");
    Layout l2 = layout_of({{8, 8}, {4, 4}, {1, 1}});
    pj::require_eq(l2.size, sizeof(S2), "S2 的大小和编译器一致");
    Layout l3 = layout_of({{1, 1}, {2, 2}, {1, 1}, {8, alignof(long long)}, {4, 4}});
    pj::require_eq(show(l3.offsets), show({offsetof(S3, a), offsetof(S3, b), offsetof(S3, c), offsetof(S3, d), offsetof(S3, e)}),
                   "S3 的偏移和编译器一致");
    pj::require_eq(l3.size, sizeof(S3), "S3 的大小和编译器一致");
    pj::require_eq(l3.align, alignof(S3), "S3 的对齐和编译器一致");
    Layout e = layout_of({});
    pj::require(e.size == 1 && e.align == 1 && e.offsets.empty(), "空结构体：大小 1、对齐 1");
  });
  pj::run("best_order_is_optimal", [] {
    std::mt19937 rng(7);
    const std::size_t aligns[] = {1, 2, 4, 8, 16};
    for (int t = 0; t < 300; ++t) {
      std::vector<Field> fs(1 + rng() % 6);
      for (Field& f : fs) {
        f.align = aligns[rng() % 5];
        f.size = f.align * (1 + rng() % 3);
      }
      std::vector<std::size_t> order = best_order(fs);
      std::vector<std::size_t> sorted = order;
      std::sort(sorted.begin(), sorted.end());
      std::vector<std::size_t> id(fs.size());
      std::iota(id.begin(), id.end(), 0);
      pj::require(sorted == id, "best_order 要返回一个排列");
      std::vector<Field> picked;
      for (std::size_t i : order) picked.push_back(fs[i]);
      std::size_t best = SIZE_MAX;
      std::vector<std::size_t> perm = id;
      do {
        std::vector<Field> p;
        for (std::size_t i : perm) p.push_back(fs[i]);
        best = std::min(best, layout_of(p).size);
      } while (std::next_permutation(perm.begin(), perm.end()));
      pj::require_eq(layout_of(picked).size, best, "best_order 得到的大小等于所有排列里最小的");
    }
    pj::require_eq(show(best_order({{2, 2}, {8, 8}, {2, 2}, {8, 8}})), std::string("{1, 3, 0, 2}"), "对齐相同时保持原顺序");
  });
  pj::run("padded_counter", [] {
    pj::require(alignof(PaddedCounter) == 64 && sizeof(PaddedCounter) == 64, "每个计数器独占 64 字节（alignof 和 sizeof 都是 64）");
    std::vector<PaddedCounter> counters(4);
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t)
      ts.emplace_back([&counters, t] {
        for (int i = 0; i < 100000; ++i) counters[t].value.fetch_add(1, std::memory_order_relaxed);
      });
    for (auto& th : ts) th.join();
    std::uint64_t sum = 0;
    for (auto& c : counters) sum += c.value.load();
    pj::require_eq(sum, std::uint64_t(400000), "4 个线程各自计数");
    auto a = reinterpret_cast<std::uintptr_t>(&counters[0]), b = reinterpret_cast<std::uintptr_t>(&counters[1]);
    pj::require(a % 64 == 0 && b - a == 64, "vector 里的计数器按缓存行对齐、相距 64 字节");
  });
}
