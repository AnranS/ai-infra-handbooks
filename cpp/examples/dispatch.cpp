#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

struct bf16 {
  std::uint16_t bits;
};
inline float to_float(float x) { return x; }
inline float to_float(bf16 x) { return std::bit_cast<float>(std::uint32_t(x.bits) << 16); }

enum class DType { F32, BF16 };

template <class T>
struct type_tag {
  using type = T;
};

// 一个按数据类型和头维度特化的 "kernel"：对一个头做归约
template <class T, int D>
float reduce_head(const void* x) {
  const T* p = static_cast<const T*>(x);
  float s = 0;
  for (int i = 0; i < D; ++i) s += to_float(p[i]);
  return s;
}

template <class F>
decltype(auto) dispatch_dtype(DType t, F&& f) {
  switch (t) {
    case DType::F32: return f(type_tag<float>{});
    case DType::BF16: return f(type_tag<bf16>{});
  }
  throw std::invalid_argument("未知的数据类型");
}

template <class F>
decltype(auto) dispatch_head_dim(int head_dim, F&& f) {
  switch (head_dim) {
    case 64: return f(std::integral_constant<int, 64>{});
    case 128: return f(std::integral_constant<int, 128>{});
    case 256: return f(std::integral_constant<int, 256>{});
    default: throw std::invalid_argument("不支持的 head_dim：" + std::to_string(head_dim));
  }
}

float run(DType dtype, int head_dim, const void* x) {
  return dispatch_dtype(dtype, [&](auto tag) {
    using T = typename decltype(tag)::type;
    return dispatch_head_dim(head_dim, [&](auto hd) {
      constexpr int D = decltype(hd)::value;   // 在这里 D 是真正的编译期常量
      return reduce_head<T, D>(x);
    });
  });
}

int main() {
  float xf[256];
  bf16 xb[256];
  for (int i = 0; i < 256; ++i) {
    xf[i] = 0.5f;
    xb[i] = bf16{0x3f80};   // bf16 的 1.0
  }
  std::printf("f32  head_dim=128 → %.1f\n", run(DType::F32, 128, xf));
  std::printf("bf16 head_dim=64  → %.1f\n", run(DType::BF16, 64, xb));
  try {
    run(DType::F32, 96, xf);
  } catch (const std::invalid_argument& e) {
    std::printf("%s\n", e.what());
  }
}
