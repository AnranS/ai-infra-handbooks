// nvcc -std=c++17 -O2 -I../../python/minisgl/kernel/csrc test_kv_kernels.cu -o test_kv_kernels
// 独立的自检程序：用随机数据分别测试两个 kernel 的向量化路径（uint4）和逐元素路径，与 CPU 结果逐位比较。
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kv_kernels.cuh"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

template <typename V>
bool test_store(int pool_tokens, int n, int row_bytes) {
  const int row = row_bytes / sizeof(V);
  std::vector<V> kc(pool_tokens * row), vc(pool_tokens * row), k(n * row), v(n * row);
  std::vector<int32_t> idx(n);
  std::vector<int> perm(pool_tokens);
  for (int i = 0; i < pool_tokens; ++i) perm[i] = i;
  for (int i = pool_tokens - 1; i > 0; --i) std::swap(perm[i], perm[rand() % (i + 1)]);
  for (int i = 0; i < n; ++i) idx[i] = perm[i];  // 互不相同的目标位置
  auto fill = [](std::vector<V>& a) { memset(a.data(), 0, a.size() * sizeof(V));
    for (size_t i = 0; i < a.size() * sizeof(V); ++i) reinterpret_cast<uint8_t*>(a.data())[i] = rand(); };
  fill(kc); fill(vc); fill(k); fill(v);
  V *dkc, *dvc, *dk, *dv; int32_t* didx;
  CHECK(cudaMalloc(&dkc, kc.size() * sizeof(V))); CHECK(cudaMalloc(&dvc, vc.size() * sizeof(V)));
  CHECK(cudaMalloc(&dk, k.size() * sizeof(V))); CHECK(cudaMalloc(&dv, v.size() * sizeof(V)));
  CHECK(cudaMalloc(&didx, n * sizeof(int32_t)));
  CHECK(cudaMemcpy(dkc, kc.data(), kc.size() * sizeof(V), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dvc, vc.data(), vc.size() * sizeof(V), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dk, k.data(), k.size() * sizeof(V), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dv, v.data(), v.size() * sizeof(V), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(didx, idx.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
  const int threads = 128, blocks = (n * 32 + threads - 1) / threads;
  minisgl::store_kv_kernel<V><<<blocks, threads>>>(dkc, dvc, didx, dk, dv, n, row, row, row);
  CHECK(cudaGetLastError());
  std::vector<V> out_k(kc.size()), out_v(vc.size());
  CHECK(cudaMemcpy(out_k.data(), dkc, kc.size() * sizeof(V), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(out_v.data(), dvc, vc.size() * sizeof(V), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {  // CPU 参考结果
    memcpy(&kc[idx[i] * row], &k[i * row], row * sizeof(V));
    memcpy(&vc[idx[i] * row], &v[i * row], row * sizeof(V));
  }
  bool ok = memcmp(out_k.data(), kc.data(), kc.size() * sizeof(V)) == 0 &&
            memcmp(out_v.data(), vc.data(), vc.size() * sizeof(V)) == 0;
  cudaFree(dkc); cudaFree(dvc); cudaFree(dk); cudaFree(dv); cudaFree(didx);
  return ok;
}

bool test_embedding(int vocab, int start, int length, int n, int dim) {
  const int row = dim * 2 / 16;  // bf16 行，按 uint4 搬运
  std::vector<uint4> w(length * row), out(n * row);
  for (auto& x : w) x = uint4{(unsigned)rand(), (unsigned)rand(), (unsigned)rand(), (unsigned)rand()};
  std::vector<int32_t> ids(n);
  for (int i = 0; i < n; ++i) ids[i] = rand() % vocab;
  uint4 *dw, *dout; int32_t* dids;
  CHECK(cudaMalloc(&dw, w.size() * sizeof(uint4))); CHECK(cudaMalloc(&dout, out.size() * sizeof(uint4)));
  CHECK(cudaMalloc(&dids, n * sizeof(int32_t)));
  CHECK(cudaMemcpy(dw, w.data(), w.size() * sizeof(uint4), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dids, ids.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
  const int threads = 128, blocks = (n * 32 + threads - 1) / threads;
  minisgl::embedding_kernel<uint4><<<blocks, threads>>>(dout, dw, dids, n, row, start, length);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(out.data(), dout, out.size() * sizeof(uint4), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < n && ok; ++i) {
    const int local = ids[i] - start;
    for (int j = 0; j < row; ++j) {
      uint4 e = (local >= 0 && local < length) ? w[local * row + j] : uint4{0, 0, 0, 0};
      uint4 g = out[i * row + j];
      ok &= e.x == g.x && e.y == g.y && e.z == g.z && e.w == g.w;
    }
  }
  cudaFree(dw); cudaFree(dout); cudaFree(dids);
  return ok;
}

int main() {
  srand(42);
  bool all = true;
  auto report = [&](const char* name, bool ok) { printf("%-40s %s\n", name, ok ? "PASS" : "FAIL"); all &= ok; };
  report("store_kv uint4  (8 heads x 128 x bf16)", test_store<uint4>(300, 37, 8 * 128 * 2));
  report("store_kv uint16 (row = 6 bytes)", test_store<uint16_t>(64, 17, 6));
  report("embedding, full vocab", test_embedding(1000, 0, 1000, 23, 64));
  report("embedding, vocab shard [500, 750)", test_embedding(1000, 500, 250, 41, 64));
  return all ? 0 : 1;
}
