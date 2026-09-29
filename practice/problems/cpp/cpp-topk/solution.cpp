#include <algorithm>
#include <numeric>
#include <span>
#include <vector>

std::vector<int> topk(std::span<const float> logits, int k) {
  if (k <= 0) return {};
  std::vector<int> idx(logits.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::size_t kk = std::min<std::size_t>(k, idx.size());
  auto cmp = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
  if (kk < idx.size()) std::nth_element(idx.begin(), idx.begin() + kk, idx.end(), cmp);
  idx.resize(kk);
  std::sort(idx.begin(), idx.end(), cmp);
  return idx;
}
