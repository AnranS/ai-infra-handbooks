---
title: 索引与掩码：padding mask、取最后一个 token、top-k 过滤
chapter: indexing.md
difficulty: 中等
tags: [masked_fill, gather, scatter_, topk, 变长批]
requires: [torch]
---
变长批次里最常写的四段代码，都和「按下标取数」有关。`lengths` 是形状 `(B,)` 的整数张量，表示每条序列的真实长度：

1. `pad_mask(lengths, max_len)`：返回 `(B, max_len)` 的布尔张量，有效位置为 `True`。不许用 Python 循环；
2. `mask_scores(scores, lengths)`：`scores` 形状 `(B, T, S)`，把**被 padding 的 key 位置**（第三维）填成 `-inf`，返回新张量，不修改输入；
3. `last_token(h, lengths)`：`h` 形状 `(B, T, D)`，取出每条序列最后一个有效位置的向量，返回 `(B, D)`。用 `gather`，不要用循环；
4. `top_k_filter(logits, k)`：`logits` 形状 `(B, V)`，每一行只保留最大的 `k` 个，其余填 `-inf`，返回新张量。用 `topk` + `scatter_`。

```python
pad_mask(torch.tensor([2, 3]), 4)
# tensor([[ True,  True, False, False],
#         [ True,  True,  True, False]])
```

<!-- 题解 -->
`pad_mask` 的标准写法是一行广播比较：`torch.arange(max_len) < lengths[:, None]`。把「每个位置的下标」和「每条序列的长度」摆成能广播的形状，比较就得到掩码 —— 这类「用 arange 造下标再比较」的套路在因果掩码、块掩码里会反复出现。

`mask_scores` 用 `masked_fill`：`scores.masked_fill(~mask[:, None, :], float("-inf"))`。注意掩码要加在**第三维**（key 的位置）上，所以是 `[:, None, :]`；加错成 `[:, :, None]` 就变成屏蔽 query，结果全是 NaN。还要注意 `masked_fill` 返回新张量、`masked_fill_` 原地改 —— 题目要求不改输入，用前者。

`last_token` 的下标是 `lengths - 1`，但 `gather` 要求下标张量和源张量**维数相同**，所以要先 `view(B, 1, 1)` 再 `expand(B, 1, D)`，取完再 `squeeze(1)`。写成 `h[torch.arange(B), lengths - 1]` 也对（高级索引），两种都接受。

`top_k_filter` 先 `vals, idx = logits.topk(k, dim=-1)`，再造一个全 `-inf` 的张量 `scatter_(-1, idx, vals)` 把保留的值放回去。反过来「把不要的位置填 -inf」要先算补集，更麻烦。采样时的 top-k 就是这么做的：填 `-inf` 而不是直接删掉，是为了让后面的 softmax 自然给出 0 概率，同时保持形状不变、能继续批处理。
