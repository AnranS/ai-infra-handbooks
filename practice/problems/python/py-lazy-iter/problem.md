---
title: 惰性的分块、滑窗与多路归并
chapter: core/iterators.md
difficulty: 中等
tags: [生成器, itertools, 惰性求值]
---
写三个**生成器函数**，它们都要能处理无限长的输入（只在需要时才从输入里取元素）：

1. `chunked(iterable, n)`：每 `n` 个元素打包成一个 `tuple`，最后一组可以不满 `n` 个。
2. `sliding_window(iterable, n)`：长度为 `n` 的滑动窗口（`tuple`）；输入不足 `n` 个元素时什么也不产生。
3. `merge_sorted(*iterables)`：把若干个**已经升序**的可迭代对象合并成一个升序序列（像 `heapq.merge`，但不要直接调用它）。

```python
list(chunked(range(7), 3))              # [(0, 1, 2), (3, 4, 5), (6,)]
list(sliding_window("abcd", 2))         # [('a', 'b'), ('b', 'c'), ('c', 'd')]
list(merge_sorted([1, 4, 9], [2, 3], [5]))   # [1, 2, 3, 4, 5, 9]
```

`n <= 0` 时 `chunked` 和 `sliding_window` 抛出 `ValueError`（在**调用时**就抛，而不是第一次 `next` 时）。

<!-- 题解 -->
- `chunked`：`it = iter(iterable)`，循环里 `chunk = tuple(itertools.islice(it, n))`，空了就结束。
- `sliding_window`：用 `collections.deque(maxlen=n)`，先填满 `n` 个，之后每来一个元素 `append` 一次、产出一次。
- `merge_sorted`：堆里放 `(当前值, 序号, 迭代器)`，序号用来在值相等时打破平局（迭代器之间不能比较大小）。
- "调用时就抛异常"：生成器函数的函数体要到第一次 `next` 才执行，所以要写成普通函数做参数检查，再返回一个内部生成器。
