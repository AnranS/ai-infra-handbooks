#include <cstddef>

struct TileConfig {
  int bm, bn, bk, stages;
};
constexpr std::size_t smem_bytes(TileConfig c, std::size_t elem) {
  return (std::size_t(c.bm) * c.bk + std::size_t(c.bk) * c.bn) * elem * c.stages;
}

template <TileConfig C, class T>
struct GemmKernel {
  static_assert(smem_bytes(C, sizeof(T)) <= 228 * 1024, "tile 太大：共享内存放不下");
};

int main() { GemmKernel<TileConfig{256, 256, 64, 4}, float> k; (void)k; }
