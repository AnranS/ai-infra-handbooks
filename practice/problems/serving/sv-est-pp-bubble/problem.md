---
title: 估算：流水线并行的气泡
chapter: distributed/pp-cp.md
difficulty: 简单
tags: [估算, 流水线并行, 气泡, 1F1B]
---
流水线并行把模型按层切成 `p` 段，一个 batch 拆成 `m` 个 micro-batch 依次流过。开头"灌满"和结尾"排空"时总有 stage 空闲，这就是**气泡**。
设每个 micro-batch 在每个 stage 上的前向 + 反向时间相同：

- GPipe 和 1F1B 的气泡时间相同：占总时间的比例是 $\frac{p-1}{m+p-1}$，相对理想计算时间的开销是 $\frac{p-1}{m}$；
- 交错式 1F1B（每张卡放 `v` 个不连续的模型块）把气泡缩小到原来的 $1/v$：比例 $\frac{p-1}{v\,m+p-1}$，开销 $\frac{p-1}{v\,m}$；
- 两者的区别在显存：GPipe 要同时保存全部 `m` 个 micro-batch 的激活，1F1B 在第一个 stage 最多只保存 `min(p, m)` 个。

实现：

1. `bubble_fraction(p, m, v=1)`：气泡占总时间的比例；
2. `bubble_overhead(p, m, v=1)`：气泡时间 ÷ 理想计算时间；
3. `min_microbatches(p, target, v=1)`：让 `bubble_fraction` 不超过 `target` 的最小 `m`；
4. `peak_inflight(p, m, schedule)`：`schedule` 为 `"gpipe"` 或 `"1f1b"`，返回第一个 stage 同时保存激活的 micro-batch 数上限。

```python
bubble_fraction(8, 32)          # 7/39，约 18%
bubble_fraction(8, 32, v=2)     # 7/71，约 9.9%
min_microbatches(8, 0.1)        # 63
```

<!-- 题解 -->
要把气泡压到 10% 以下，`m` 至少要约 $9(p-1)$。micro-batch 数又受全局 batch 大小限制（全局 batch = DP × m × micro-batch 大小），
所以 PP 越深，要么全局 batch 越大，要么用交错调度（代价是更多的点对点通信），要么用零气泡调度（把反向拆成对输入的梯度和对权重的梯度，用后者填气泡）。

推理里的流水线并行没有反向，气泡问题在连续批处理下基本不存在，主要代价变成单请求延迟增加（要依次经过每一段）。
