---
title: 估算：滚动发布与扩容的时间账
chapter: k8s/deploy-scale.md
difficulty: 中等
tags: [估算,滚动更新,扩缩容,K8s]
---
推理服务的启动和退出都很慢，发布和扩容的时间要提前算出来。实现：

1. `rollout_batches(replicas, max_surge, max_unavailable)`：滚动更新一共要分几批（每批能同时替换 `max_surge + max_unavailable` 个，至少 1 个，最后一批可以不满）；
2. `rollout_seconds(replicas, ready_s, drain_s, max_surge, max_unavailable)`：总耗时。每一批的耗时是 `max(ready_s, drain_s)`（新副本启动与旧副本退出并行），总耗时 = 批数 × 每批耗时；
3. `extra_gpus(replicas, gpus_per_replica, max_surge)`：发布期间需要的额外 GPU 数；
4. `capacity_loss(replicas, max_unavailable)`：发布期间最少可用的副本比例（0～1）；
5. `scale_lag_seconds(ready_s, image_pull_s=0, node_provision_s=0)`：一次扩容从触发到新副本可用的总延迟。

```python
rollout_batches(12, 2, 1)                       # 4
rollout_seconds(12, 150, 60, 2, 1)              # 600
extra_gpus(12, 8, 2)                            # 16
round(capacity_loss(12, 1), 3)                  # 0.917
scale_lag_seconds(150, image_pull_s=120, node_provision_s=180)   # 450
```

<!-- 题解 -->
三个数字决定发布窗口：**副本数**、**新副本就绪时间**、**每批能换几个**。推理服务的 `ready_s` 常常是几分钟（拉镜像、加载几十 GB 权重、编译 CUDA Graph），所以 12 个副本、150 秒就绪、每批换 3 个时，整个发布要 10 分钟——这还是在一切顺利的前提下。

`max_surge` 的代价是真金白银的 GPU：`maxSurge=2` 且每副本 8 张卡，发布期间就要多准备 16 张。`max_unavailable` 的代价是容量：发布期间最少只有 11/12 的容量。GPU 服务往往两者都紧张，所以实践中要么在低峰期发布，要么先临时扩一批节点。

`scale_lag_seconds` 提醒一件容易忘的事：**扩容的延迟不只是 Pod 启动**。如果集群里没有空闲节点，还要等云厂商开机器（几分钟）和拉镜像（几十 GB）。这就是为什么推理服务要预留缓冲容量、用低优先级占位 Pod 预热、以及把镜像提前分发到节点。
