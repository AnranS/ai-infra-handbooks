#include <cstddef>
#include <cstdio>

struct TileConfig {
  int bm, bn, bk, stages;
};

constexpr std::size_t smem_bytes(TileConfig c, std::size_t elem) {
  return (std::size_t(c.bm) * c.bk + std::size_t(c.bk) * c.bn) * elem * c.stages;
}
constexpr std::size_t kSmemPerSm = 228 * 1024;   // H100 每个 SM 的共享内存

template <class... Dims>
constexpr std::size_t numel(Dims... d) { return (std::size_t(d) * ... * 1); }   // 折叠表达式

template <TileConfig C, class T>   // C++20：结构体也可以当非类型模板参数
struct GemmKernel {
  static constexpr std::size_t smem = smem_bytes(C, sizeof(T));
  static_assert(smem <= kSmemPerSm, "tile 太大：共享内存放不下");
  static_assert(C.bm % 16 == 0 && C.bn % 16 == 0, "tile 边长必须是 16 的倍数（Tensor Core 的要求）");
  static constexpr int blocks_per_sm = int(kSmemPerSm / (smem + 1024));
};

int main() {
  using K1 = GemmKernel<TileConfig{128, 128, 64, 3}, short>;
  using K2 = GemmKernel<TileConfig{128, 256, 64, 4}, short>;
  static_assert(numel(2, 3, 4) == 24);
  std::printf("K1：共享内存 %zu KiB，每个 SM %d 个块\n", K1::smem / 1024, K1::blocks_per_sm);
  std::printf("K2：共享内存 %zu KiB，每个 SM %d 个块\n", K2::smem / 1024, K2::blocks_per_sm);
}
