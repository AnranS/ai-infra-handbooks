---
title: 二元语言模型与困惑度
chapter: basics/language-model.md
difficulty: 简单
tags: [语言模型, 平滑, 困惑度]
---
实现一个最简单的语言模型：二元模型（bigram），下一个词只依赖当前词。

**`BigramLM(sentences, k=1.0)`**：`sentences` 是若干个句子，每个句子是词的列表。训练时在每个句子前后加上 `"<s>"` 和 `"</s>"`，统计相邻词对的次数。
词表 `vocab` 是训练语料里出现过的所有词加上 `"</s>"`（**不含** `"<s>"`，它只作为上下文出现）和 `"<unk>"`；不在词表里的词在训练和计算时都当成 `"<unk>"`。

**`lm.prob(prev, word)`**：用 add-k 平滑计算 $P(w \mid v) = \dfrac{c(v, w) + k}{c(v) + k \cdot |V|}$，
其中 $c(v)$ 是 $v$ 作为上下文出现的次数，$|V|$ 是词表大小。

**`lm.perplexity(sentence)`**：句子（同样补上 `<s>`、`</s>`）的困惑度 $\exp\!\big(-\frac{1}{N}\sum \log P(w_i \mid w_{i-1})\big)$，$N$ 是预测的词数（含 `</s>`，不含 `<s>`）。

```python
lm = BigramLM([["我", "爱", "GPU"], ["我", "爱", "CUDA"]], k=1)
lm.prob("我", "爱")          # (2 + 1) / (2 + 1 * 6) = 0.375，词表：我 爱 GPU CUDA </s> <unk>
lm.perplexity(["我", "爱", "GPU"])
```

<!-- 题解 -->
用 `Counter` 统计 `(prev, word)` 和 `prev` 的次数；未登录词先映射成 `<unk>`。
困惑度就是"平均每个位置在多少个词里等概率地猜"，是交叉熵的指数。add-k 平滑保证没见过的词对概率也不为 0，否则困惑度是无穷大。
