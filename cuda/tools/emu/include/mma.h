// Emulator WMMA: every thread keeps a full copy of each 16x16 fragment. Loads, mma and stores are
// performed redundantly by all lanes, which is semantically equivalent for checking tiling logic.
#pragma once
#include <cuda_fp16.h>
#include <type_traits>
namespace nvcuda {
namespace wmma {
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };

template <class Use, int M, int N, int K, class T, class Layout = void>
struct fragment {
  static constexpr int R = std::is_same<Use, matrix_b>::value ? K : M;
  static constexpr int C = std::is_same<Use, matrix_a>::value ? K : N;
  float v[R][C];
};

template <class Use, int M, int N, int K, class T, class L>
void fill_fragment(fragment<Use, M, N, K, T, L>& f, float x) {
  for (auto& row : f.v) for (auto& e : row) e = x;
}
template <class Use, int M, int N, int K, class L>
void load_matrix_sync(fragment<Use, M, N, K, half, L>& f, const half* p, unsigned ldm) {
  constexpr bool rm = std::is_same<L, row_major>::value;
  for (int r = 0; r < fragment<Use, M, N, K, half, L>::R; ++r)
    for (int c = 0; c < fragment<Use, M, N, K, half, L>::C; ++c)
      f.v[r][c] = __half2float(rm ? p[r * ldm + c] : p[c * ldm + r]);
}
template <int M, int N, int K, class LA, class LB>
void mma_sync(fragment<accumulator, M, N, K, float, void>& d, const fragment<matrix_a, M, N, K, half, LA>& a,
              const fragment<matrix_b, M, N, K, half, LB>& b, const fragment<accumulator, M, N, K, float, void>& c) {
  float out[M][N];
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = c.v[i][j];
      for (int k = 0; k < K; ++k) s += a.v[i][k] * b.v[k][j];
      out[i][j] = s;
    }
  for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) d.v[i][j] = out[i][j];
}
template <int M, int N, int K>
void store_matrix_sync(float* p, const fragment<accumulator, M, N, K, float, void>& f, unsigned ldm, layout_t layout) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) (layout == mem_row_major ? p[r * ldm + c] : p[c * ldm + r]) = f.v[r][c];
}
}  // namespace wmma
}  // namespace nvcuda
