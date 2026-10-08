#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

std::vector<int> top_p(const std::vector<float>& probs, float p) {
  std::vector<int> idx(probs.size());
  std::iota(idx.begin(), idx.end(), 0);
  auto cmp = [&](int a, int b) { return probs[a] > probs[b] || (probs[a] == probs[b] && a < b); };
  for (std::size_t k = 64;; k *= 2) {
    k = std::min(k, idx.size());
    std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(), cmp);
    std::sort(idx.begin(), idx.begin() + k, cmp);
    float acc = 0;
    for (std::size_t i = 0; i < k; ++i) {
      acc += probs[idx[i]];
      if (acc >= p) return {idx.begin(), idx.begin() + i + 1};
    }
    if (k == idx.size()) return idx;   // 浮点误差导致总和略小于 p：全部保留
  }
}

int main() {
  std::vector<float> probs(1000, 0.0005f);   // 997 个小概率，合计约 0.4985
  probs[42] = 0.3f;
  probs[7] = 0.15f;
  probs[900] = 0.0515f;
  auto keep = top_p(probs, 0.5f);
  std::printf("保留 %zu 个：", keep.size());
  for (std::size_t i = 0; i < std::min<std::size_t>(keep.size(), 4); ++i) std::printf("%d ", keep[i]);
  std::printf("\n");
  std::printf("p=0.9 保留 %zu 个\n", top_p(probs, 0.9f).size());
}
