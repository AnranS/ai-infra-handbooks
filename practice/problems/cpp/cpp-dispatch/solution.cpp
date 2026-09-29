#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

struct bf16 {
  std::uint16_t bits;
};
struct f16 {
  std::uint16_t bits;
};
enum class DType { F32, BF16, F16 };

template <class T>
struct type_tag {
  using type = T;
};

template <class F>
decltype(auto) dispatch_head_dim(int head_dim, F&& f) {
  switch (head_dim) {
    case 64: return f(std::integral_constant<int, 64>{});
    case 96: return f(std::integral_constant<int, 96>{});
    case 128: return f(std::integral_constant<int, 128>{});
    case 256: return f(std::integral_constant<int, 256>{});
    default: throw std::invalid_argument("不支持的 head_dim：" + std::to_string(head_dim));
  }
}

template <class F>
decltype(auto) dispatch_dtype(DType t, F&& f) {
  switch (t) {
    case DType::F32: return f(type_tag<float>{});
    case DType::BF16: return f(type_tag<bf16>{});
    case DType::F16: return f(type_tag<f16>{});
  }
  throw std::invalid_argument("未知的数据类型");
}
