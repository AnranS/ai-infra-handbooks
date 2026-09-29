---
title: 序列打包：切样本、重置位置与文档掩码
chapter: scratch/data.md
difficulty: 中等
tags: [数据, 序列打包, 注意力掩码, varlen]
requires: [numpy]
---
预训练把很多篇文档首尾相接、切成定长的样本，避免补零浪费。但一个样本里会有几篇文档，文档之间不应该互相看见。实现：

1. `pack(docs, seq_len, eos)`：`docs` 是若干个 token 列表。每篇文档后面接一个 `eos`，全部拼成一条长序列，再切成长度 `seq_len + 1` 的样本（前 `seq_len` 个是输入，错开一位是目标）：第 `i` 个样本是长序列的 `[i·seq_len, i·seq_len + seq_len + 1)`，相邻样本共用一个 token；最后凑不满的丢掉。返回样本列表；
2. `positions(tokens, eos)`：每个 token 的位置编号。样本开头从 0 开始，**紧跟在 `eos` 后面**的 token 重新从 0 开始（`eos` 本身属于前一篇文档）；
3. `cu_seqlens(tokens, eos)`：FlashAttention 变长接口要的段边界：`[0, 第一段结束, 第二段结束, …, len(tokens)]`，每段在 `eos` 之后结束（最后一个 token 是 `eos` 时不要重复 `len(tokens)`）；
4. `doc_mask(tokens, eos)`：`(L, L)` 的布尔数组，`mask[i][j]` 为真当且仅当 `j <= i` 且 `i`、`j` 在同一段里。

```python
pack([[1, 2, 3], [4, 5], [6, 7, 8, 9]], seq_len=4, eos=0)
# 长序列 [1 2 3 0 4 5 0 6 7 8 9 0] → [[1, 2, 3, 0, 4], [4, 5, 0, 6, 7], [7, 8, 9, 0]] 里只留满的两个
# → [[1, 2, 3, 0, 4], [4, 5, 0, 6, 7]]
positions([1, 2, 3, 0, 4], 0)     # [0, 1, 2, 3, 0]
cu_seqlens([1, 2, 3, 0, 4], 0)    # [0, 4, 5]
```

<!-- 题解 -->
三样东西描述的是同一件事——样本里的"段"：`cu_seqlens` 是段的边界，`positions` 在每段开头归零（RoPE 的位置从 0 算），`doc_mask` 是块对角的因果掩码（大模型原理手册[注意力机制](llm://transformer/attention/#因果掩码)一章的交互小工具里可以看到它的样子）。用了 `cu_seqlens` 的 varlen kernel 根本不算段与段之间的块，计算量比完整的因果注意力还少。

不做文档掩码（像本书"从零训练"教程那样直接在拼接的长序列上训练）也很常见，模型会学会"看到 `eos` 就不再依赖前文"；但文档掩码能避免跨文档的虚假关联，长上下文训练时尤其重要（Llama 3 就在同一个序列里的文档之间加了掩码）。
