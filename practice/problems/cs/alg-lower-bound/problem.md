---
title: 二分查找的两个边界
chapter: algo/array-string.md
difficulty: 简单
tags: [二分,模板]
---
手写二分的两个模板（不能用 `bisect`）：

1. `lower_bound(a, x)`：升序数组里第一个 **≥ x** 的下标（都比 x 小就返回 `len(a)`）；
2. `upper_bound(a, x)`：第一个 **> x** 的下标；
3. `count_of(a, x)`：`x` 在数组里出现了几次。

```python
lower_bound([1, 2, 2, 2, 3], 2)     # 1
upper_bound([1, 2, 2, 2, 3], 2)     # 4
count_of([1, 2, 2, 2, 3], 2)        # 3
count_of([1, 3], 2)                 # 0
```

<!-- 题解 -->
统一用**半开区间** `[lo, hi)`：`lo = 0`、`hi = len(a)`，循环条件 `while lo < hi`，中点 `mid = (lo + hi) // 2`，然后二选一——`lo = mid + 1`（mid 一定不是答案）或 `hi = mid`（mid 可能是答案）。循环结束时 `lo == hi`，就是答案。

这个模板的好处是：不用纠结 `hi = len(a) - 1` 还是 `len(a)`、不用担心死循环（每轮区间至少缩小一半）、两个边界只差一个比较符号（`<` 对 `<=`）。

出现次数 = `upper_bound - lower_bound`，这也是判断"x 是否存在"的写法：`lower_bound(a, x) < len(a) and a[lower_bound(a, x)] == x`。
