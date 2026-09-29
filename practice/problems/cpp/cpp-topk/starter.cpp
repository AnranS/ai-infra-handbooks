#include <algorithm>
#include <numeric>
#include <span>
#include <vector>

std::vector<int> topk(std::span<const float> logits, int k) {
  std::vector<int> idx(logits.size());
  std::iota(idx.begin(), idx.end(), 0);
  // TODO：取最大的 k 个，按 logit 降序、相等时 id 升序
  idx.resize(std::min<std::size_t>(std::max(k, 0), idx.size()));
  return idx;
}
