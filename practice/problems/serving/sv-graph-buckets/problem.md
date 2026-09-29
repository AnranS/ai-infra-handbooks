---
title: 选 CUDA Graph 的录制批大小
chapter: engine/graphs-compile.md
difficulty: 中等
tags: [CUDA Graph, 动态规划, 补齐]
---
CUDA Graph 要为每个批大小单独录一张图，每张图都占显存；实际的 batch 要**向上补齐**到最近的已录制大小，多出来的是浪费的计算。

1. `pad_to(bs, sizes)`：`sizes` 是升序的已录制批大小；返回不小于 `bs` 的最小值；`bs` 超过最大值时返回 `None`（不能用图，走 eager 模式）；
2. `expected_waste(dist, sizes)`：`dist` 是字典 `{批大小: 出现次数}`，返回平均每步补齐浪费的请求数 $\frac{\sum_b c_b (\text{pad}(b) - b)}{\sum_b c_b}$（超过最大录制大小的批次不计入浪费，但计入分母）；
3. `best_sizes(dist, max_bs, k)`：在 `1..max_bs` 里选恰好 `k` 个批大小（**必须包含 `max_bs`**），使 `expected_waste` 最小；有多个最优解时返回字典序最小的升序列表。

```python
dist = {1: 10, 2: 5, 3: 30, 7: 25, 8: 5}
best_sizes(dist, 8, 3)     # [3, 7, 8]
```

`max_bs` 最大到 256、`k` 最大到 20，要求用动态规划，不能枚举所有组合。

<!-- 题解 -->
把选出的大小排序为 $s_1 < \dots < s_k = \text{max\_bs}$，区间 $(s_{j-1}, s_j]$ 里的批次都补齐到 $s_j$，浪费只和相邻两个选中值有关，所以可以 DP：

$$f[j][s] = \min_{t < s} \Big(f[j-1][t] + \sum_{t < b \le s} c_b (s - b)\Big)$$

区间代价用前缀和 $O(1)$ 算出：$\sum c_b (s - b) = s \cdot \sum c_b - \sum c_b b$。总复杂度 $O(k \cdot n^2)$。
要得到字典序最小的解，DP 时对相同代价保留更小的前驱（或者代价相同时比较整个方案）。

vLLM 默认录 `[1, 2, 4, 8, 16, 24, ..., 512]`，SGLang 类似：小批量间隔密（常见、而且补齐浪费的比例大），大批量间隔疏。
