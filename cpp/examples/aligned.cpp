#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct alignas(16) Float4 {
  float x, y, z, w;
};
struct alignas(64) PaddedCounter {   // 独占一个缓存行
  std::int64_t value;
};

bool aligned_to(const void* p, std::size_t a) { return reinterpret_cast<std::uintptr_t>(p) % a == 0; }

int main() {
  std::printf("alignof(Float4)=%zu sizeof(PaddedCounter)=%zu\n", alignof(Float4), sizeof(PaddedCounter));
  void* page = std::aligned_alloc(4096, 1 << 20);   // 大小必须是对齐值的倍数
  std::printf("aligned_alloc 按 4096 对齐：%s\n", aligned_to(page, 4096) ? "是" : "否");
  std::free(page);
  auto* counters = new PaddedCounter[4];            // C++17 起 new 会遵守 alignas
  std::printf("new PaddedCounter[] 按 64 对齐：%s\n", aligned_to(counters, 64) ? "是" : "否");
  delete[] counters;
}
