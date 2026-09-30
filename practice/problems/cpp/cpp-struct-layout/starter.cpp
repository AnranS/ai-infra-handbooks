#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

struct Field {
  std::size_t size, align;
};

struct Layout {
  std::vector<std::size_t> offsets;
  std::size_t size, align;
};

constexpr std::size_t align_up(std::size_t n, std::size_t a) { return n; }   // TODO

Layout layout_of(const std::vector<Field>& fields) {
  Layout out{{}, 0, 1};
  for (const Field& f : fields) {                  // TODO：没有对齐，也没有末尾的填充
    out.offsets.push_back(out.size);
    out.size += f.size;
  }
  return out;
}

std::vector<std::size_t> best_order(const std::vector<Field>& fields) {
  std::vector<std::size_t> idx(fields.size());
  std::iota(idx.begin(), idx.end(), 0);
  return idx;                                      // TODO
}

struct PaddedCounter {                             // TODO：独占一个缓存行
  std::atomic<std::uint64_t> value{0};
};
