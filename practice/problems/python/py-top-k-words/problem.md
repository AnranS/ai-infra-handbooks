---
title: 出现最多的 k 个词
chapter: core/containers.md
difficulty: 简单
tags: [Counter, heapq, 排序]
---
给定一个单词列表 `words` 和整数 `k`，返回出现次数最多的 `k` 个单词，按出现次数**从多到少**排序；次数相同时按**字典序从小到大**。

```python
top_k_words(["i", "love", "llm", "i", "love", "gpu"], 2)   # ["i", "love"]
top_k_words(["b", "a", "c", "a", "b", "c"], 2)             # ["a", "b"]（三个都出现 2 次，按字典序）
```

- `k` 保证不超过不同单词的个数；
- 单词数可能到 $10^5$ 量级，要求 $O(n \log k)$ 或 $O(n \log n)$。

<!-- 题解 -->
`collections.Counter` 计数，然后按 `(-次数, 单词)` 排序取前 `k` 个：

```python
counts = Counter(words)
return sorted(counts, key=lambda w: (-counts[w], w))[:k]
```

也可以用 `heapq.nsmallest(k, counts, key=lambda w: (-counts[w], w))`，复杂度 $O(n \log k)$。
注意 `Counter.most_common(k)` 在次数相同时按**首次出现的顺序**排，不满足"字典序"的要求。
