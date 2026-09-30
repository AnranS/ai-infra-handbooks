---
title: 估算：切卡还是独占
chapter: k8s/gpu.md
difficulty: 简单
tags: [估算,MIG,共享,K8s]
---
判断一个小模型该独占整卡、用 MIG 还是时间片共享。实现：

1. `mig_share(profile)`：返回 `(一张卡能切几份, 每份显存 GB, 每份算力/带宽占比)`。支持 H100 80GB 的四个档位：`1g.10gb`、`2g.20gb`、`3g.40gb`、`7g.80gb`（份数分别按 1/2/3/7 个切片算，一张卡共 7 个切片）；
2. `fits(profile, weight_gb, kv_gb)`：权重加上 KV 能不能放进一份；
3. `decode_ms(weight_gb, bandwidth_gbs, share=1.0)`：batch=1 的 decode 下限（读一遍权重），`share` 是带宽份额；
4. `choose(weight_gb, kv_gb, latency_budget_ms, tenants, bandwidth_gbs=3350)`：在 `1g.10gb`、`2g.20gb`、`3g.40gb`、`7g.80gb` 里，选**能装下且满足延迟预算、并且一张卡能切出至少 `tenants` 份**的最省档位（切得越多越省）；都不满足返回 `None`。

```python
mig_share("2g.20gb")                       # (3, 20, 2/7)
fits("1g.10gb", 3, 4)                      # True
round(decode_ms(3, 3350, 1/7), 2)          # 6.27
choose(3, 4, 30, tenants=5)                # "1g.10gb"
choose(14, 6, 30, tenants=3)               # "2g.20gb"
```

<!-- 题解 -->
MIG 把**显存和带宽一起切**，所以判断分两步：先看"权重 + KV 放不放得进这一份的显存"，再看"按份额打折后的 decode 延迟还满不满足预算"。

例子里 3 GB 的小模型可以切到最小的 `1g.10gb`，一张卡服务 7 个租户，每个租户的 decode 下限 6.3 ms——对多数场景完全够用，卡的利用率提高了好几倍。而 14 GB 的 7B 模型最小只能用 `2g.20gb`（10 GB 装不下），一张卡切 3 份。

时间片共享是另一条路：不切显存、延迟随租户数线性变差、没有隔离。它适合开发环境；生产的多租户要用 MIG，因为一个租户的 OOM 不该影响别人。

真实决策还要考虑：MIG 切分需要重置整张卡（不能在线改）、部分特性（NVLink、MPS）在 MIG 下受限、以及能不能干脆把多个租户合并到同一个推理实例上（通常这才是最省卡的做法）。
