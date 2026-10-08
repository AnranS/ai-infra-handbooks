#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

std::vector<int> topk(const std::vector<float>& logits, int k) {
  std::vector<int> idx(logits.size());
  std::iota(idx.begin(), idx.end(), 0);
  auto cmp = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
  std::nth_element(idx.begin(), idx.begin() + k, idx.end(), cmp);   // O(n)：前 k 个就是最大的 k 个，但无序
  idx.resize(k);
  std::sort(idx.begin(), idx.end(), cmp);                            // 只排这 k 个
  return idx;
}

int main() {
  std::vector<float> logits(32000);
  for (int i = 0; i < 32000; ++i) logits[i] = float((i * 7919) % 32000) / 32000.0f;
  for (int id : topk(logits, 5)) std::printf("%d:%.5f ", id, logits[id]);
  std::printf("\n");
}
