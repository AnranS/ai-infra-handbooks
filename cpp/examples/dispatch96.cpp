#include <cstdio>
#include <stdexcept>
#include <type_traits>

template <int D>
float reduce_head(const float* x) {
  static_assert(D % 8 == 0, "head_dim 必须是 8 的倍数：kernel 按 8 个元素一组做向量化加载");
  float s = 0;
  for (int i = 0; i < D; ++i) s += x[i];
  return s;
}

template <class F>
decltype(auto) dispatch_head_dim(int head_dim, F&& f) {
  switch (head_dim) {
    case 64: return f(std::integral_constant<int, 64>{});
    case 96: return f(std::integral_constant<int, 96>{});
    case 128: return f(std::integral_constant<int, 128>{});
    case 256: return f(std::integral_constant<int, 256>{});
    default: throw std::invalid_argument("不支持的 head_dim");
  }
}

int main() {
  float x[256];
  for (auto& v : x) v = 1.0f;
  for (int hd : {64, 96, 128}) {
    float r = dispatch_head_dim(hd, [&](auto d) { return reduce_head<decltype(d)::value>(x); });
    std::printf("head_dim=%d → %.0f\n", hd, r);
  }
}
