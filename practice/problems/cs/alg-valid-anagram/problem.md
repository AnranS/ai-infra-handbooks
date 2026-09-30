---
title: 字母异位词
chapter: algo/array-string.md
difficulty: 简单
tags: [哈希表,计数,字符串]
---
判断两个字符串是否是字母异位词（字符种类和个数完全相同，顺序可以不同）。另外实现一个分组函数。

1. `is_anagram(s, t)`：布尔值；
2. `group_anagrams(words)`：把互为异位词的分到一组，返回 `list[list[str]]`；组内保持输入顺序，组之间按**每组第一个词在输入里的顺序**排列。

```python
is_anagram("anagram", "nagaram")                # True
is_anagram("rat", "car")                        # False
group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
# [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

<!-- 题解 -->
判断用 `Counter(s) == Counter(t)`（O(n)），或者排序后比较（O(n log n)）。长度不同可以先快速排除。

分组的关键是设计一个**规范形式**（canonical form）当哈希键：把每个词排序后的字符串（`"eat" -> "aet"`），或者 26 个字母的计数元组。前者简单，后者对长词更快。用 `dict` 保持插入顺序（Python 3.7+ 保证），就能满足"组之间按第一次出现的顺序"。

这是"用哈希表按某种等价关系分组"的模板：等价类的规范形式是什么，就用它当键。推理系统里的对应：前缀缓存把"一串 token"哈希成块的键、按模型与参数把请求分组、按 LoRA 适配器分桶。
