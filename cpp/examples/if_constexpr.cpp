#include <bit>
#include <concepts>
#include <cstdint>
#include <cstdio>

struct bf16 {
  std::uint16_t bits;
};

template <class T>
concept Scalar = std::floating_point<T> || std::same_as<T, bf16>;

template <Scalar T, int D>
float sum_head(const T* x) {
  float s = 0;
  if constexpr (std::same_as<T, bf16>) {
    for (int i = 0; i < D; ++i) s += std::bit_cast<float>(std::uint32_t(x[i].bits) << 16);   // 先转 float 再累加
  } else {
    for (int i = 0; i < D; ++i) s += static_cast<float>(x[i]);
  }
  if constexpr (D % 8 != 0) {
    std::printf("（D=%d 不是 8 的倍数：真实 kernel 在这里走不了 128 位的向量化加载）\n", D);
  }
  return s;
}

int main() {
  float f[96];
  bf16 b[64];
  for (auto& v : f) v = 0.25f;
  for (auto& v : b) v = bf16{0x4000};   // bf16 的 2.0
  std::printf("%.1f\n", sum_head<float, 96>(f));
  std::printf("%.1f\n", sum_head<bf16, 64>(b));
  std::printf("%.1f\n", sum_head<float, 20>(f));
}
