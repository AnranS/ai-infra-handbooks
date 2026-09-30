---
title: 最长无重复子串
chapter: algo/array-string.md
difficulty: 中等
tags: [滑动窗口,哈希表,字符串]
---
求字符串里**没有重复字符**的最长子串的长度。

```python
longest_unique("abcabcbb")      # 3："abc"
longest_unique("bbbbb")         # 1
longest_unique("pwwkew")        # 3："wke"
```

<!-- 题解 -->
滑动窗口的标准题。窗口里维护"每个字符最后出现的位置"，右指针每前进一格：如果这个字符上次出现的位置在窗口内（`>= left`），就把左边界跳到"上次出现位置 + 1"。

跳转比一格一格收缩更快，也更不容易写错：左边界只会往右走，永远不要让它回退（`left = max(left, last[ch] + 1)` 里的 `max` 就是干这个的）。

时间 O(n)，空间 O(字符集大小)。变体：最多含 k 种字符的最长子串（窗口里维护计数，超过 k 种就收缩）、最多把 k 个 0 变成 1 的最长全 1 段。
