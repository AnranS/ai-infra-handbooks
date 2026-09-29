---
title: 采样前的 top-k
chapter: basics/stl-perf.md
difficulty: 简单
tags: [标准库算法, nth_element, 确定性]
---
实现 `std::vector<int> topk(std::span<const float> logits, int k)`：返回 logits 最大的 `k` 个 token 的 id，**按 logit 从大到小排列，logit 相等时 id 小的在前**（采样结果要可复现，顺序必须是确定的）。

- `k <= 0` 返回空；`k` 大于词表大小时返回全部 id（同样按上面的顺序）；
- 词表可能有十几万个 token，而 `k` 通常只有几十：不要对整个词表排序。

```cpp
std::vector<float> logits = {0.1f, 2.0f, -1.0f, 2.0f, 0.5f};
topk(logits, 3);   // {1, 3, 4}：两个 2.0 按 id 排，再是 0.5
```

<!-- 题解 -->
先用 `std::nth_element` 把最大的 `k` 个放到前面（平均 O(n)），再只对这 `k` 个 `sort`。比较函数必须是**严格弱序**：`logits[a] > logits[b] || (logits[a] == logits[b] && a < b)`，
相等时按 id 决胜负，保证结果确定。见 [标准库的性能视角](cpp://basics/stl-perf/) 的"算法"一节。
