#include <array>
#include <cstdio>

template <class T>
T dot(const T* a, const T* b, int n) {
  T s{};
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

// HEAD_DIM 是编译期常量：数组大小固定、循环次数已知，编译器可以完全展开、向量化
template <int HEAD_DIM>
float qk_score(const float* q, const float* k) {
  std::array<float, HEAD_DIM> prod{};
  for (int i = 0; i < HEAD_DIM; ++i) prod[i] = q[i] * k[i];
  float s = 0;
  for (float p : prod) s += p;
  return s;
}

int main() {
  float q[128], k[128];
  double qd[4] = {1, 2, 3, 4};
  for (int i = 0; i < 128; ++i) {
    q[i] = 1.0f;
    k[i] = 0.5f;
  }
  std::printf("dot<float>=%.1f dot<double>=%.1f\n", dot(q, k, 128), dot(qd, qd, 4));
  std::printf("qk_score<64>=%.1f qk_score<128>=%.1f\n", qk_score<64>(q, k), qk_score<128>(q, k));
}
