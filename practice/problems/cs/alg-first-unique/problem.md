---
title: 第一个不重复的元素
chapter: algo/linked-stack-hash.md
difficulty: 简单
tags: [哈希表,计数,顺序]
---
实现两个函数：

1. `first_unique_char(s)`：返回字符串里第一个只出现一次的字符的下标，没有返回 -1；
2. `top_k_frequent(nums, k)`：返回出现次数最多的 k 个元素，按**次数从多到少**；次数相同时按**第一次出现的顺序**。

```python
first_unique_char("leetcode")          # 0
first_unique_char("loveleetcode")      # 2
top_k_frequent([1, 1, 1, 2, 2, 3], 2)  # [1, 2]
top_k_frequent([4, 4, 5, 5, 6], 2)     # [4, 5]
```

<!-- 题解 -->
两题都是"计数 + 保序"。`Counter` 在 Python 3.7+ 保持插入顺序，所以第一次遍历建的计数表，遍历它就是按第一次出现的顺序。

`first_unique_char`：先数一遍，再按原顺序找第一个计数为 1 的。两遍 O(n)，不要写成"对每个字符再数一遍"的 O(n²)。

`top_k_frequent`：`Counter.most_common(k)` 就能做，但它在次数相同时的顺序依赖实现细节；要显式保证"次数相同按第一次出现"，就自己排序：`sorted(cnt, key=lambda x: -cnt[x])`——Python 的排序是稳定的，所以相同次数时保持 `cnt` 的插入顺序，正好就是第一次出现的顺序。

k 远小于元素种类数时，更优的做法是用大小为 k 的最小堆（O(n log k)）或桶排序（O(n)），见[排序、堆与贪心](sort-heap-greedy.md)。
