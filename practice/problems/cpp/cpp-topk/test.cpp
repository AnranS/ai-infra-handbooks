#include <algorithm>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

static std::string show(const std::vector<int>& v) {
  std::string s = "{";
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
  return s + "}";
}

static std::vector<int> reference(const std::vector<float>& l, int k) {
  std::vector<int> idx(l.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
  idx.resize(std::clamp<long>(k, 0, long(idx.size())));
  return idx;
}

int main() {
  pj::run("example", [] {
    std::vector<float> logits = {0.1f, 2.0f, -1.0f, 2.0f, 0.5f};
    pj::require_eq(show(topk(logits, 3)), std::string("{1, 3, 4}"), "top-3");
  });
  pj::run("edge_k", [] {
    std::vector<float> logits = {3, 1, 2};
    pj::require_eq(show(topk(logits, 0)), std::string("{}"), "k=0");
    pj::require_eq(show(topk(logits, -5)), std::string("{}"), "k<0");
    pj::require_eq(show(topk(logits, 10)), std::string("{0, 2, 1}"), "k 大于词表：返回全部并排序");
    pj::require_eq(show(topk(std::vector<float>{}, 3)), std::string("{}"), "空词表");
  });
  pj::run("ties_are_deterministic", [] {
    std::vector<float> logits(1000, 1.0f);   // 全部相等
    logits[777] = 2.0f;
    pj::require_eq(show(topk(logits, 5)), std::string("{777, 0, 1, 2, 3}"), "相等时按 id 从小到大");
  });
  pj::run("random_vs_reference", [] {
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> dist(-50, 50);   // 取值范围小：大量相等的 logit
    for (int trial = 0; trial < 20; ++trial) {
      std::vector<float> logits(5000);
      for (auto& x : logits) x = float(dist(rng)) * 0.5f;
      int k = 1 + trial * 13;
      pj::require(topk(logits, k) == reference(logits, k), "与参考实现不一致（k=" + std::to_string(k) + "）");
    }
  });
}
