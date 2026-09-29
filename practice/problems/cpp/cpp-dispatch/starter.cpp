#include <cstdint>
#include <stdexcept>
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
  (void)head_dim;
  return f(std::integral_constant<int, 64>{});   // TODO
}

template <class F>
decltype(auto) dispatch_dtype(DType t, F&& f) {
  (void)t;
  return f(type_tag<float>{});   // TODO
}
