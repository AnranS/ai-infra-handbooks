---
title: gang 调度与 GPU 碎片
chapter: k8s/multi-node.md
difficulty: 中等
tags: [Kubernetes,gang 调度,多机多卡,碎片]
---
多机多卡的任务必须"要么全部调度、要么一个都不调度"。节点是 `{"name": ..., "free": 空闲卡数}` 的列表。实现：

1. `place_group(nodes, members, gpus_each)`：给一个任务组的 `members` 个成员各分配 `gpus_each` 张卡（一个成员的卡必须来自同一个节点），返回 `[(节点名, 卡数), ...]` 并扣减 `free`；放不下就**全部回滚**并返回 `None`。分配时优先选空闲卡最少但仍放得下的节点（装箱，尽量保住整机）；并列时按名字字典序；
2. `fragmentation(nodes, gpus_each)`：返回 `(空闲总数, 可用于该规格任务的卡数)`——后者是每个节点能整除出的部分之和；
3. `defrag_gain(nodes, gpus_each)`：如果把所有零散任务重新整理（假设空闲卡可以任意集中），能多放下几个成员。

```python
nodes = [{"name": "a", "free": 2}, {"name": "b", "free": 2}, {"name": "c", "free": 4}]
place_group(nodes, members=2, gpus_each=2)     # [("a", 2), ("b", 2)]
fragmentation(nodes, 4)                        # (8, 4)
defrag_gain(nodes, 4)                          # 1
```

<!-- 题解 -->
`place_group` 的关键是**整组回滚**：一个成员放不下时，前面已经占用的卡必须全部还回去，否则会出现"占着一半资源等另一半"的死锁——这正是 Volcano 的 PodGroup、Kueue 的 Workload 在做的事。

装箱式的选点（优先用剩得最少但够用的节点）是为了**保住整机**：单卡任务如果均匀撒开，8 卡任务就再也放不下了。

`fragmentation` 量化了这件事：三个节点分别剩 2、2、4 张，一共 8 张，但要跑 4 卡任务时只有 4 张可用（只有节点 c 能整除出一份）。`defrag_gain` 告诉你重新整理能多放几个——这就是 descheduler 和"定期重调度"存在的理由。注意真实系统里还有拓扑约束（同一个成员的卡要 NVLink 相连），只会更严格。
