#include <concepts>
#include <cstdint>

struct bf16 {
  std::uint16_t bits;
};
template <class T>
concept Scalar = std::floating_point<T> || std::same_as<T, bf16>;

template <Scalar T>
float first(const T* x) { return static_cast<float>(x[0]); }

int main() {
  int tokens[4] = {1, 2, 3, 4};
  return static_cast<int>(first(tokens));   // int 不是 Scalar
}
