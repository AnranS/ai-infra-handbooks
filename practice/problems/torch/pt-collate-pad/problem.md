---
title: collate_fn：变长样本补齐、长度分桶与 padding 占比
chapter: training.md
difficulty: 简单
tags: [DataLoader, collate_fn, padding, 分桶]
requires: [torch]
---
默认的 collate 只会 `torch.stack`，长度不一样就直接报错。自己写三个函数：

1. `collate(batch)`：`batch` 是 `[(ids, label), ...]` 的列表，`ids` 是一维 `int64` 张量、长度各不相同，`label` 是 `int`。返回一个 dict：
   - `ids`：`(B, Lmax)`，右侧补 `0`，`Lmax` 是**这一批**里最长的；
   - `mask`：`(B, Lmax)` 的布尔张量，有效位置为 `True`；
   - `labels`：`(B,)` 的 `int64` 张量。
2. `bucket_batches(lengths, batch_size)`：`lengths` 是一个 Python 列表。先按长度**从短到长**排序，再按 `batch_size` 切块，返回「下标列表」的列表（最后一块可以不满）；
3. `pad_fraction(lengths, batches)`：给定分好的批，返回 padding 占总格子数的比例（`float`）。每一批占 `len(batch) * max(该批长度)` 个格子，其中有效的是各长度之和。

```python
bucket_batches([5, 1, 3, 2], 2)   # [[1, 3], [2, 0]]
```

<!-- 题解 -->
**补齐到「这一批」的最长，而不是全局最长**。全局补齐写起来最省事，但浪费的算力随最长样本增长；按批补齐几乎不要钱，是所有变长任务的默认做法。补完一定要把 `mask` 一起返回 —— padding 位置既不能参与注意力，也不能算进 loss，后面要靠它。

补什么值不重要（这里用 0），重要的是别让它影响结果。右侧补齐时因果注意力天然保证 padding 影响不到它前面的真 token，所以不需要额外的 attention mask；**左侧补齐就需要**，这是两种做法最大的区别。

**分桶**就是让长度接近的样本进同一批。排序后切块是最简单的实现，`pad_fraction` 能直接量出收益：随机分批时长样本会把整批撑大，分桶之后每批的最大值和平均值很接近，浪费的格子立刻掉下来；真实的变长文本上常常能省掉三成以上的计算。代价是打乱程度下降，工程上的折中是「桶内随机 + 桶间随机」，而不是严格按长度排。

这三个函数合起来就是 `DataLoader(dataset, batch_sampler=..., collate_fn=...)` 的两个钩子：`batch_sampler` 决定**谁和谁一批**，`collate_fn` 决定**怎么拼成张量**。
