#include <cstdio>
#include <vector>

struct Counts {
  int copies = 0, moves = 0;
};

template <bool Noexcept>
struct Item {
  static inline Counts c;
  Item() = default;
  Item(const Item&) { ++c.copies; }
  Item(Item&&) noexcept(Noexcept) { ++c.moves; }
};

template <bool N>
void run(const char* name) {
  std::vector<Item<N>> v;
  for (int i = 0; i < 5; ++i) v.push_back(Item<N>{});   // 容量 1 → 2 → 4 → 8，扩容 3 次
  std::printf("%s：copies=%d moves=%d\n", name, Item<N>::c.copies, Item<N>::c.moves);
}

int main() {
  run<false>("移动构造没有 noexcept");
  run<true>("移动构造标了 noexcept");
}
