---
title: 最长回文子串
chapter: algo/array-string.md
difficulty: 中等
tags: [字符串,中心扩展,动态规划]
---
1. `longest_palindrome(s)`：返回最长的回文**子串**（连续）。有多个同样长的返回**最靠左**的；
2. `count_palindromes(s)`：统计回文子串的个数（位置不同算不同，单个字符也算）。

```python
longest_palindrome("babad")      # "bab"
longest_palindrome("cbbd")       # "bb"
count_palindromes("aaa")         # 6
```

<!-- 题解 -->
**中心扩展**：一个回文串由它的中心向两边对称展开，中心可能是一个字符（奇数长度）或两个字符之间的缝（偶数长度），一共 `2n - 1` 个中心。对每个中心尽量往两边扩，时间 O(n²)、空间 O(1)。

统计个数用同一套循环：每成功扩展一次就是一个新的回文子串。

动态规划的写法（`f[i][j]` 表示 `s[i..j]` 是否回文）也是 O(n²)，但空间是 O(n²)，还要注意遍历顺序（按长度从小到大）。中心扩展更简洁。

真正 O(n) 的算法是 **Manacher**，面试里一般只要求能说出"存在线性算法"即可。
