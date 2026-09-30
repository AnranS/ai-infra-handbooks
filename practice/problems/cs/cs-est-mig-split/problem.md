---
title: 估算：一张卡切几份划算
chapter: arch/multi-gpu.md
difficulty: 简单
tags: [估算, MIG, decode, 容量规划]
---
MIG 把一张 GPU 在硬件上切成几份，SM、显存和**显存带宽**都按份切开。一张 H100（132 个 SM、80 GB、3350 GB/s）按"份数 / 7"切分：一个 `slices` 份的实例有 `slices / 7` 的带宽、`round(132 * slices / 7)` 个 SM，显存按档位给定。实现：

1. `mig_profile(slices, mem_gb, sms=132, bw_gbs=3350)`：返回 `{"sms": ..., "mem_gb": ..., "bw_gbs": ...}`（SM 数四舍五入到整数）；
2. `decode_ms(weight_gb, profile)`：这份实例上 batch 为 1 的 decode 一步下限（毫秒），即读一遍权重；放不下权重（`weight_gb` 超过显存）时返回 `None`；
3. `best_split(weight_gb, budget_ms, options)`：`options` 是 `[(名称, slices, mem_gb), ...]`。在满足 `decode_ms <= budget_ms` 的档位里，选**一张卡能切出最多实例**的那个（`7 // slices`）；并列时选 slices 小的；都不满足返回 `None`。返回 `(名称, 实例数)`。

```python
p = mig_profile(3, 40)                       # {"sms": 57, "mem_gb": 40, "bw_gbs": 1435.7...}
round(decode_ms(14, p), 1)                   # 9.8
best_split(3, 30, [("1g.10gb", 1, 10), ("2g.20gb", 2, 20), ("3g.40gb", 3, 40)])   # ("1g.10gb", 7)
```

<!-- 题解 -->
MIG 常被误解成"只切算力"，其实带宽也一起切。decode 是带宽受限的，所以一份 `3g.40gb` 的 decode 速度只有整卡的 43%（3/7），而不是"SM 少了一点"。

判断该不该切，就按"模型放得下吗、切完的带宽还够不够达到延迟目标"两步算。小模型（1.5B 的 BF16 权重 3 GB）在 `1g.10gb` 上读一遍权重只要 6.3 ms，离 30 ms 的目标还很远，切 7 份完全可行，这正是 MIG 适合的场景：很多小任务、要求互相隔离。大模型反过来：70B 的权重 140 GB，任何一份都放不下，只能用整卡甚至多卡。
