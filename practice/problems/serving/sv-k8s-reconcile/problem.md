---
title: 控制器的 reconcile 循环
chapter: k8s/basics.md
difficulty: 中等
tags: [Kubernetes,控制器,幂等,reconcile]
---
实现一个 Deployment 控制器的 reconcile 逻辑。集群状态用字典表示：

```python
state = {"spec": {"replicas": 3, "image": "v2"},
         "pods": {"p1": {"image": "v1", "ready": True}, "p2": {"image": "v2", "ready": True}}}
```

- `reconcile(state, max_surge=1, max_unavailable=1)`：返回这一轮要做的**一个**动作，格式是 `("create", 镜像)`、`("delete", pod 名)` 或 `None`（已收敛）。规则按优先级：
  1. 可用副本数（`ready` 且镜像是最新的）不足 `replicas - max_unavailable` 且总数未超过 `replicas + max_surge` 时，创建一个新镜像的 Pod；
  2. 总数超过 `replicas + max_surge`，或者总数已达 `replicas` 且还有旧镜像的 Pod 时，删掉一个**旧镜像**的 Pod（按名字排序取第一个）；
  3. 总数少于 `replicas`，创建一个新镜像的 Pod；
  4. 总数多于 `replicas`，删掉一个 Pod（优先旧镜像，其次按名字排序的最后一个）；
  5. 否则返回 `None`。
- `run(state, max_steps=100, **kw)`：反复 reconcile 并施加动作（新建的 Pod 名字是 `pod-1`、`pod-2`…… 按创建顺序递增，创建时 `ready=False`，每一步开始时把所有 Pod 置为 `ready=True`，模拟它们启动完成），返回执行过的动作列表。

```python
s = {"spec": {"replicas": 2, "image": "v1"}, "pods": {}}
run(s)                      # [("create", "v1"), ("create", "v1")]
len(s["pods"])              # 2
```

<!-- 题解 -->
这道题把"滚动更新"和"扩缩容"统一成同一个循环：每次只根据**当前状态**决定一个动作，不记忆历史。这正是 Kubernetes 控制器的工作方式，也是它能随时重启、能自愈的原因。

三个容易写错的地方：（1）**可用副本**要同时满足 `ready` 和"镜像是最新的"，否则滚动更新时会把旧 Pod 算成可用而提前删光；（2）**动作只做一个**就返回，让调用方重新观测状态，避免基于过期信息连续操作；（3）**幂等**：收敛之后再调用必须返回 `None`，否则控制器会永远忙个不停（真实系统里表现为 API server 被刷爆）。

`max_surge` 和 `max_unavailable` 决定了滚动更新的节奏：前者是"最多可以多出几个"（要额外的 GPU），后者是"最多可以少几个"（牺牲容量）。
