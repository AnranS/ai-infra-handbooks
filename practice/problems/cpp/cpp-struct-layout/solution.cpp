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

constexpr std::size_t align_up(std::size_t n, std::size_t a) { return (n + a - 1) & ~(a - 1); }

Layout layout_of(const std::vector<Field>& fields) {
  Layout out{{}, 0, 1};
  std::size_t off = 0;
  for (const Field& f : fields) {
    off = align_up(off, f.align);
    out.offsets.push_back(off);
    off += f.size;
    out.align = std::max(out.align, f.align);
  }
  out.size = fields.empty() ? 1 : align_up(off, out.align);
  return out;
}

std::vector<std::size_t> best_order(const std::vector<Field>& fields) {
  std::vector<std::size_t> idx(fields.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return fields[a].align > fields[b].align; });
  return idx;
}

struct alignas(64) PaddedCounter {
  std::atomic<std::uint64_t> value{0};
};
