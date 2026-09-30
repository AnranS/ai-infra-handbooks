---
title: 配额、优先级与抢占
chapter: k8s/operator-ops.md
difficulty: 简单
tags: [Kubernetes,配额,优先级,抢占]
---
实现多租户集群的准入与抢占逻辑：

1. `admit(quota, used, request)`：三个字典的键相同（如 `{"cpu": ..., "gpu": ...}`）。返回 `(是否允许, 原因)`：`request` 里有 `quota` 没声明的资源时返回 `(False, "未知资源")`；任何一项 `used + request > quota` 时返回 `(False, "超配额")`；否则 `(True, "")`。特别地，`request` 里缺少 `quota` 声明的项时返回 `(False, "缺少 requests")`（开了配额就必须写 requests）；
2. `preempt(pods, need_gpu, my_priority)`：`pods` 是 `[{"name":..., "priority":..., "gpu":...}, ...]`，从**优先级低于** `my_priority` 的 Pod 里挑一组驱逐，使释放的 GPU 数 ≥ `need_gpu`。优先驱逐优先级最低的；同优先级时先驱逐占卡多的（尽量少动几个 Pod）；凑不够返回 `None`；
3. `gpu_hours(pods, hours)`：按优先级分组统计 GPU 小时数，返回 `{优先级: GPU 小时}`。

```python
admit({"gpu": 8}, {"gpu": 6}, {"gpu": 1})          # (True, "")
admit({"gpu": 8}, {"gpu": 6}, {"gpu": 4})          # (False, "超配额")
preempt([{"name": "b1", "priority": 100, "gpu": 2}], 2, 1000)   # ["b1"]
```

<!-- 题解 -->
`admit` 对应 ResourceQuota 的准入逻辑，两个容易忽略的点：开了配额之后**不写 requests 的 Pod 会被直接拒绝**（真实报错是 `must specify requests.cpu`），以及配额里没声明的资源类型不受限制但也不该被请求（生产上通常配合 LimitRange 给默认值）。

`preempt` 对应调度器的抢占：只能抢**优先级更低**的 Pod，并且要挑一组代价最小的受害者。这里的规则是"先抢优先级最低的、同优先级先抢占卡多的"，这样驱逐的 Pod 数量最少。真实调度器还要考虑 PDB、抢占后该节点是否真的放得下（抢占是按节点做的），以及被抢占 Pod 的优雅退出时间。

`gpu_hours` 是配额审计的基本单位：按优先级（或团队、队列）统计 GPU 小时，才能回答"谁在用卡""在线和离线各占多少"。
