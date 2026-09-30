---
title: 单词拆分
chapter: algo/dp-backtrack.md
difficulty: 中等
tags: [动态规划,字符串,记忆化]
---
1. `word_break(s, words)`：字符串 `s` 能否被拆成字典 `words` 里的若干个单词（每个单词可重复使用）；
2. `min_cuts(s, words)`：最少拆成几段；拆不了返回 -1。

```python
word_break("leetcode", ["leet", "code"])     # True
word_break("catsandog", ["cats", "dog", "sand", "and", "cat"])   # False
min_cuts("catsanddog", ["cat", "cats", "and", "sand", "dog"])    # 3
```

<!-- 题解 -->
状态是"前 i 个字符能否被拆分"：`f[0] = True`，`f[i] = any(f[j] and s[j:i] in words)`。把 `words` 放进 `set` 让查找 O(1)，并且可以用"最长单词长度"限制 `j` 的范围，把复杂度从 O(n²·L) 降到 O(n·maxlen)。

`min_cuts` 把布尔换成整数：`g[i] = min(g[j] + 1)`，无解用无穷大表示。

朴素的回溯（枚举每个切点）在"aaaa…a"配上 `["a", "aa"]` 这种输入上会指数爆炸，加记忆化就变成了上面的 DP——这是"暴力递归 → 记忆化 → 递推"路径的又一个例子。

这道题在推理里有个真实对应：分词（tokenizer）的最优切分。BPE 是贪心合并，而 Unigram 模型用的正是 Viterbi 动态规划，在所有切分中找概率最大的那个（见[分词](llm://basics/tokenization/)）。
