---
title: 从零训练 BPE 分词器
chapter: basics/tokenization.md
difficulty: 困难
tags: [BPE, 分词, 贪心]
---
实现经典的字节对编码（BPE，Sennrich 2016 的词级版本）。

**`train_bpe(corpus, num_merges)`**：`corpus` 是若干行文本。

1. 按空白切成单词，统计每个单词出现的次数；
2. 每个单词表示成字符序列，末尾加一个词尾符号 `"</w>"`：`"low"` → `("l", "o", "w", "</w>")`；
3. 重复 `num_merges` 次：统计所有相邻符号对的出现次数（按单词出现次数加权），选出**次数最多**的一对；次数相同时选**字典序最小**的（比较 `(a, b)` 元组）；把所有单词里的这一对合并成一个新符号 `a + b`（从左到右，不重叠）；
4. 没有任何相邻符号对时提前停止。

返回合并规则的列表 `[(a, b), ...]`，按学到的顺序。

**`encode(word, merges)`**：把一个单词切成子词：从字符序列（加 `"</w>"`）开始，反复找出**当前相邻符号对里规则排名最靠前**（在 `merges` 里下标最小）的那一对并合并它的所有出现，直到没有可以合并的对。

```python
corpus = ["low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3]
merges = train_bpe(corpus, 10)
merges[:3]                     # [('e', 's'), ('es', 't'), ('est', '</w>')]
encode("lowest", merges)       # ['low', 'est</w>']
```

<!-- 题解 -->
训练时维护"单词（符号元组）→ 次数"的字典。每一轮：

1. 遍历每个单词的相邻对，累加 `pairs[(a, b)] += count`；
2. `best = min(pairs, key=lambda p: (-pairs[p], p))` 同时实现"次数最多、字典序最小"；
3. 用一个 `merge(symbols, pair)` 函数从左到右扫描，遇到 `(a, b)` 就合并并跳过两个位置。

编码时给每条规则一个排名 `rank = {pair: i}`，每次在当前序列的相邻对里取排名最小的合并，这和训练时的合并顺序一致。
真实的分词器（GPT-2、LLaMA 3）用字节级 BPE，思路完全相同，只是初始符号是 256 个字节。
