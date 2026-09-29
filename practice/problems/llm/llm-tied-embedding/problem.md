---
title: 嵌入层、共享权重的输出层与 padding
chapter: transformer/embedding.md
difficulty: 简单
tags: [Embedding, 权重共享, 批处理]
---
小模型常把输入嵌入矩阵 $E \in \mathbb{R}^{V \times d}$ 和输出层共享（`tie_word_embeddings`）：logits $= h E^\top$。
实现一个 `TiedLM`，用 numpy 完成嵌入查表、输出层、以及一批变长序列的 padding 处理：

- `TiedLM(E)`：保存嵌入矩阵；
- `embed(ids)`：`ids` 是整数数组（任意形状），返回形状 `ids.shape + (d,)` 的嵌入；id 越界（`< 0` 或 `>= V`）抛出 `IndexError`；
- `logits(h)`：`h` 形状 `(..., d)`，返回 `(..., V)`；
- `pad_batch(seqs, pad_id)`：把若干个长度不同的 token 列表**右侧**补齐到最长长度，返回 `(ids, mask)`：`ids` 形状 `(B, T)` 的 int64 数组，`mask` 形状 `(B, T)` 的 bool 数组，真实 token 处为 `True`；
- `last_token_logits(seqs, pad_id)`：对每个序列，取**最后一个真实 token** 的嵌入，经过输出层，返回形状 `(B, V)` 的 logits（这里"隐藏状态"就直接用嵌入，省去中间的层）。

```python
E = np.random.randn(10, 4)
lm = TiedLM(E)
ids, mask = lm.pad_batch([[1, 2, 3], [4]], pad_id=0)   # ids=[[1,2,3],[4,0,0]]
lm.last_token_logits([[1, 2, 3], [4]], pad_id=0)       # 第一行是 E[3] @ E.T，第二行是 E[4] @ E.T
```

<!-- 题解 -->
查表就是花式索引 `E[ids]`，numpy 自动得到 `ids.shape + (d,)`；负数下标在 numpy 里是合法的（从末尾数），所以要自己检查越界。

最后一个真实 token 的位置是 `mask.sum(1) - 1`，用 `ids[np.arange(B), lengths - 1]` 取出来。
右侧 padding 时如果直接取 `ids[:, -1]`，短序列取到的是 pad——这是批量推理里最常见的 bug 之一。
