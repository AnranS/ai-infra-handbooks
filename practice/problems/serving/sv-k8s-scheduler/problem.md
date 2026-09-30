---
title: 迷你调度器：过滤与打分
chapter: k8s/scheduling.md
difficulty: 中等
tags: [Kubernetes,调度器,GPU,装箱]
---
实现一个简化的 Kubernetes 调度器。节点和 Pod 都是字典：

```python
node = {"name": "gpu-a", "cpu": 96, "mem": 512, "gpu": 8,
        "labels": {"gpu": "h100"}, "taints": ["maintenance"],
        "used": {"cpu": 0, "mem": 0, "gpu": 0}}
pod = {"cpu": 8, "mem": 64, "gpu": 1, "selector": {"gpu": "h100"}, "tolerations": []}
```

1. `filter_nodes(nodes, pod)`：返回 `(可行节点列表, {节点名: 淘汰原因})`。依次检查资源是否足够（`"资源不足"`）、`selector` 里的标签是否全部匹配（`"标签不匹配"`）、节点的污点是否都被容忍（`"污点未容忍"`），第一条不满足就记该原因；
2. `score(node, pod, strategy)`：`"spread"` 返回放下这个 Pod 之后三种资源剩余比例的平均值 ×100（没有 GPU 的节点不计 GPU 这一项），`"binpack"` 返回 `100 -` 该值；
3. `schedule(nodes, pod, strategy="spread")`：返回选中的节点名（并把资源记到 `used` 上），没有可行节点返回 `None`。分数相同时按节点名字典序取小的。

```python
schedule(nodes, {"cpu": 8, "mem": 64, "gpu": 1, "selector": {"gpu": "h100"}})   # "gpu-a"
```

<!-- 题解 -->
这就是 kube-scheduler 的两段式：**过滤**（predicate，把放不下的节点淘汰掉并记录原因——`kubectl describe pod` 里 `FailedScheduling` 那行就是这些原因的汇总）和**打分**（score，在可行节点里挑最优）。

两种打分策略的区别很重要：`spread`（默认的 LeastAllocated）把负载摊开，单点故障影响小；`binpack`（MostAllocated）把任务挤在少数节点上，**空出整机**给多卡任务用。GPU 集群通常要后者，否则单卡任务撒满所有节点后，8 卡任务永远排不上——这就是碎片化。

实现时注意：打分要算"**放下这个 Pod 之后**"的剩余比例，而不是当前剩余；没有 GPU 的节点不能把 GPU 项算进平均值（除零）。
