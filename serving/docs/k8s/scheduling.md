# 调度器：从 filter/score 到 GPU 与 gang 调度

<p class="lead">"为什么我的推理 Pod 一直 Pending"是平台岗最常被问到的问题，答案几乎总在调度器里：资源不够、标签不匹配、污点没容忍、或者整机碎片化到放不下一个 8 卡任务。这一章先把默认调度器的两段式流程（过滤 + 打分）写成一个能跑的迷你调度器，再讲推理负载特有的三件事——GPU 是不可压缩的整数资源、多机多卡需要 gang 调度、碎片化会让"还剩很多卡"和"放不下任务"同时成立——最后给出扩展调度器的几条路子。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 默认调度器为一个 Pod 选节点分哪两步？各自在做什么？
    2. `LeastAllocated` 和 `MostAllocated` 两种打分策略各适合什么场景？
    3. 为什么多机多卡的任务需要 gang 调度？没有它会怎样？
    4. 集群里还剩 8 张卡，为什么一个 4 卡任务放不下？
    5. Pending 的 Pod，`describe` 里该看哪一行？

??? success "自测参考答案（先自己答，再展开对照）"
    1. **过滤（filter / predicate）**：逐个节点判断"放不放得下"——资源是否够（按 requests 算）、nodeSelector/亲和性是否匹配、污点是否被容忍、端口是否冲突等，不满足的直接淘汰。**打分（score）**：给剩下的节点打分再取最高分，默认综合了资源均衡、镜像本地性、拓扑分散等插件。
    2. `LeastAllocated`（默认）把负载摊开，适合在线服务，单点故障影响小、突发有余量；`MostAllocated` 是装箱，把任务塞满少数节点，好处是空出整机供大任务使用、也利于缩容省钱，适合离线任务和 GPU 集群。
    3. 多机多卡的任务只有全部成员都起来才能开始（要建立通信组），一个成员拿不到卡，其他成员就白占着卡等——多个这样的任务互相卡住就是死锁。gang 调度（Volcano、Kueue、YuniKorn）保证"要么全部调度、要么一个都不调度"。
    4. 因为卡分散在多个节点上：比如三个节点各剩 2、2、4 张，加起来 8 张，但没有一个节点能一次给出 4 张（假设任务要求同机）。这就是碎片化，缓解办法是装箱式打分、按卡数分池、以及定期做整理（重调度）。
    5. `Events` 里的 `FailedScheduling`，它会明确写出每个节点被淘汰的原因，例如 `0/1 nodes are available: 1 Insufficient nvidia.com/gpu`。

## 两段式：先过滤，再打分

![图：kube-scheduler 的两段式——过滤、打分、绑定](../assets/figures/k8s-filter-score.svg){.aig-svg}

kube-scheduler 的核心循环是：从队列里取一个未调度的 Pod → 过滤出可行节点 → 给可行节点打分 → 选最高分 → 绑定（写 `pod.spec.nodeName`）。把它写出来：

```python title="sched.py"
"""一个迷你 Kubernetes 调度器：过滤（predicate）+ 打分（score）+ 绑定，外加 gang 调度。

和真实调度器的对应：filter 对应 NodeResourcesFit、NodeAffinity、TaintToleration 等插件；
score 对应 NodeResourcesFit（LeastAllocated / MostAllocated）、ImageLocality、PodTopologySpread 等。
"""


class Node:
    def __init__(self, name, cpu, mem_gb, gpus, labels=None, taints=()):
        self.name, self.cpu, self.mem, self.gpus = name, cpu, mem_gb, gpus
        self.labels = labels or {}
        self.taints = set(taints)
        self.used_cpu = self.used_mem = self.used_gpu = 0

    def free(self):
        return self.cpu - self.used_cpu, self.mem - self.used_mem, self.gpus - self.used_gpu

    def place(self, pod):
        self.used_cpu += pod["cpu"]
        self.used_mem += pod["mem"]
        self.used_gpu += pod["gpu"]

    def release(self, pod):
        self.used_cpu -= pod["cpu"]
        self.used_mem -= pod["mem"]
        self.used_gpu -= pod["gpu"]


def filters(node, pod):
    """预选：任何一条不满足就淘汰这个节点，返回 (是否可行, 原因)"""
    cpu, mem, gpu = node.free()
    if pod["cpu"] > cpu or pod["mem"] > mem or pod["gpu"] > gpu:
        return False, "资源不足"
    for key, value in pod.get("selector", {}).items():          # nodeSelector / nodeAffinity
        if node.labels.get(key) != value:
            return False, f"标签不匹配 {key}={value}"
    if node.taints - set(pod.get("tolerations", ())):            # 污点与容忍
        return False, "有未被容忍的污点"
    return True, ""


def score_least_allocated(node, pod):
    """默认策略：剩余比例越高分越高（把负载摊开）"""
    cpu, mem, gpu = node.free()
    parts = [(cpu - pod["cpu"]) / node.cpu, (mem - pod["mem"]) / node.mem]
    if node.gpus:
        parts.append((gpu - pod["gpu"]) / node.gpus)
    return sum(parts) / len(parts) * 100


def score_most_allocated(node, pod):
    """装箱策略：剩余比例越低分越高（把碎片攒到一起，利于腾出整机）"""
    return 100 - score_least_allocated(node, pod)


def schedule(nodes, pod, score=score_least_allocated):
    """返回 (选中的节点, 各节点的分数或淘汰原因)"""
    feasible, report = [], {}
    for node in nodes:
        ok, why = filters(node, pod)
        if ok:
            feasible.append(node)
            report[node.name] = round(score(node, pod), 1)
        else:
            report[node.name] = why
    if not feasible:
        return None, report
    best = max(feasible, key=lambda n: (report[n.name], n.name))
    best.place(pod)
    return best, report


def schedule_gang(nodes, pods, score=score_least_allocated):
    """gang 调度：要么全部放下，要么一个都不放（避免多机多卡任务占着资源互相等待）"""
    placed = []
    for pod in pods:
        node, _ = schedule(nodes, pod, score)
        if node is None:
            for n, p in placed:                                  # 回滚已经放下的
                n.release(p)
            return None
        placed.append((node, pod))
    return [(n.name, p["name"]) for n, p in placed]


def fragmentation(nodes, gpus_per_pod):
    """碎片：加起来还剩很多卡，但没有一个节点能放下一个完整的任务"""
    free_total = sum(n.free()[2] for n in nodes)
    placeable = sum(n.free()[2] // gpus_per_pod for n in nodes)
    return free_total, placeable * gpus_per_pod
```

```python
from sched import (Node, fragmentation, schedule, schedule_gang, score_least_allocated,
                   score_most_allocated)


def cluster():
    return [Node("gpu-a", 96, 512, 8, {"gpu": "h100"}),
            Node("gpu-b", 96, 512, 8, {"gpu": "h100"}),
            Node("gpu-c", 64, 256, 4, {"gpu": "a100"}, taints=("maintenance",)),
            Node("cpu-d", 64, 256, 0)]


pod = {"name": "vllm-0", "cpu": 8, "mem": 64, "gpu": 1, "selector": {"gpu": "h100"}}
nodes = cluster()
node, report = schedule(nodes, pod)
print("选中：", node.name)
for name, value in report.items():
    print(f"  {name}: {value}")

print("\n同一批请求，两种打分策略下的分布：")
for name, score in (("摊开（LeastAllocated）", score_least_allocated), ("装箱（MostAllocated）", score_most_allocated)):
    nodes = cluster()
    for i in range(6):
        schedule(nodes, {"name": f"p{i}", "cpu": 8, "mem": 64, "gpu": 1, "selector": {"gpu": "h100"}}, score)
    print(f"  {name}：", {n.name: n.used_gpu for n in nodes if n.gpus})

print("\ngang 调度（一个任务要 4 张卡，且必须同时放下）：")
nodes = cluster()
for i in range(12):                                   # 先用零散的单卡任务把集群打散
    schedule(nodes, {"name": f"s{i}", "cpu": 4, "mem": 16, "gpu": 1})
print("  当前每个节点剩余 GPU：", {n.name: n.free()[2] for n in nodes if n.gpus})
small_gang = [{"name": f"train-{i}", "cpu": 8, "mem": 32, "gpu": 2} for i in range(2)]
print("  两个 2 卡的成员：", schedule_gang(nodes, small_gang))
for n in nodes:                                       # 撤销上一次，换一个更大的 gang 再试
    n.used_cpu = n.used_mem = 0
    n.used_gpu = {"gpu-a": 6, "gpu-b": 6, "gpu-c": 0}.get(n.name, 0)
big_gang = [{"name": f"big-{i}", "cpu": 8, "mem": 32, "gpu": 3} for i in range(2)]
print("  两个 3 卡的成员：", schedule_gang(nodes, big_gang), "（放不下时整组回滚，不占资源）")
print("  回滚之后每个节点剩余 GPU：", {n.name: n.free()[2] for n in nodes if n.gpus})
total, usable = fragmentation(nodes, 4)
print(f"  集群还剩 {total} 张卡，但能凑出完整 4 卡任务的只有 {usable} 张——这就是碎片")

print("\n没有节点满足时给出原因（排障时 describe 里看到的就是这些）：")
nodes = cluster()
big = {"name": "huge", "cpu": 8, "mem": 64, "gpu": 16, "selector": {"gpu": "h100"}}
node, report = schedule(nodes, big)
print("  结果：", node)
for name, value in report.items():
    print(f"  {name}: {value}")
```

```text title="输出"
选中： gpu-b
  gpu-a: 88.9
  gpu-b: 88.9
  gpu-c: 标签不匹配 gpu=h100
  cpu-d: 资源不足

同一批请求，两种打分策略下的分布：
  摊开（LeastAllocated）： {'gpu-a': 3, 'gpu-b': 3, 'gpu-c': 0}
  装箱（MostAllocated）： {'gpu-a': 0, 'gpu-b': 6, 'gpu-c': 0}

gang 调度（一个任务要 4 张卡，且必须同时放下）：
  当前每个节点剩余 GPU： {'gpu-a': 2, 'gpu-b': 2, 'gpu-c': 4}
  两个 2 卡的成员： [('gpu-b', 'train-0'), ('gpu-a', 'train-1')]
  两个 3 卡的成员： None （放不下时整组回滚，不占资源）
  回滚之后每个节点剩余 GPU： {'gpu-a': 2, 'gpu-b': 2, 'gpu-c': 4}
  集群还剩 8 张卡，但能凑出完整 4 卡任务的只有 4 张——这就是碎片

没有节点满足时给出原因（排障时 describe 里看到的就是这些）：
  结果： None
  gpu-a: 资源不足
  gpu-b: 资源不足
  gpu-c: 资源不足
  cpu-d: 资源不足
```

对着输出看四件事：

- **过滤给出的是"为什么不行"**，这正是 `kubectl describe pod` 里 `FailedScheduling` 那行的内容。
- **打分策略决定分布形态**：同样 6 个单卡请求，`LeastAllocated` 摊成 3 + 3，`MostAllocated` 挤成 6 + 0。GPU 集群通常更想要后者——空出整机才能接住 8 卡的大任务。
- **gang 调度要么全给要么不给**：两个 3 卡成员放不下时整组回滚，不会留下"占着 3 张卡等另一半"的半吊子状态。
- **碎片是"总量够但放不下"**：还剩 8 张卡，能凑出完整 4 卡任务的只有 4 张。

真实调度器的插件框架有更多扩展点（`PreFilter`、`Filter`、`PostFilter`、`PreScore`、`Score`、`Reserve`、`Permit`、`Bind`），常用插件包括 `NodeResourcesFit`（资源）、`NodeAffinity`（亲和）、`TaintToleration`（污点）、`PodTopologySpread`（跨可用区/机架分散）、`InterPodAffinity`（与某些 Pod 靠近或远离）、`ImageLocality`（镜像已在本地）。

## GPU 为什么特殊

| | CPU / 内存 | GPU（`nvidia.com/gpu`） |
| --- | --- | --- |
| 可压缩 | CPU 可以（超了被节流） | 不可以 |
| 可小数 | CPU 可以（`500m`） | 不可以，只能整数 |
| requests / limits | 可以不等 | 必须相等（写 limits 即可） |
| 超卖 | 常见 | 默认不允许 |
| 来源 | kubelet 直接上报 | **device plugin** 上报（下一章展开） |

由此产生三个后果：

1. **一张卡只能给一个容器**（除非用 MIG 切分或时间片共享，见下一章）。所以"半张卡的小模型"这种需求在 Kubernetes 原生模型里是表达不出来的。
2. **碎片化是 GPU 集群的常态**。一个 8 卡节点被 3 个单卡任务占了 3 张，剩下 5 张就接不住 8 卡任务了。缓解手段：打分用装箱、按卡数分池（1 卡池 / 2 卡池 / 整机池）、给大任务预留节点（`node taint` + 专用队列）、以及定期重调度（descheduler）把零散任务挪到一起。
3. **拓扑比数量更重要**。同一个节点上的 8 张卡也分 NVLink 域和 PCIe 拓扑，多卡任务要拿到"互相之间 NVLink 直连"的那几张卡，否则通信慢一个数量级。这需要 NVIDIA 的 GPU Operator 加拓扑感知的分配策略（或者 DRA，见下一章）。

在没有 GPU 的集群上申请一张卡，得到的就是这样的事件：

```bash
kubectl apply -f pending.yaml    # Pod 里写了 limits: {nvidia.com/gpu: 1}
kubectl describe pod needs-gpu | sed -n '/Events/,$p'
```

```text title="输出（本机示例）"
Events:
  Type     Reason            Age   From               Message
  ----     ------            ----  ----               -------
  Warning  FailedScheduling  6s    default-scheduler  0/1 nodes are available: 1 Insufficient nvidia.com/gpu. preemption: 0/1 nodes are available: 1 No preemption victims found for incoming pod.
```

这行信息量很大：**几个节点里有几个可用**、**被淘汰的原因分别是什么**、**抢占能不能救**（`No preemption victims found` 表示连抢占低优先级 Pod 都腾不出资源）。

## 把 Pod 引导到正确的节点

```yaml
spec:
  nodeSelector:                       # 最简单：必须匹配的标签
    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
  tolerations:                        # 容忍 GPU 节点上的污点（GPU 节点通常打污点防止普通 Pod 挤进来）
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  affinity:
    nodeAffinity:                     # 更灵活：硬约束 + 软偏好
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - {key: node.kubernetes.io/instance-type, operator: In, values: [p5.48xlarge, p4d.24xlarge]}
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
              - {key: topology.kubernetes.io/zone, operator: In, values: [cn-north-1a]}
    podAntiAffinity:                  # 同一个服务的副本尽量分散到不同节点
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          podAffinityTerm:
            labelSelector: {matchLabels: {app: vllm}}
            topologyKey: kubernetes.io/hostname
  topologySpreadConstraints:          # 更精细的分散：每个可用区之间副本数差不超过 1
    - maxSkew: 1
      topologyKey: topology.kubernetes.io/zone
      whenUnsatisfiable: ScheduleAnyway
      labelSelector: {matchLabels: {app: vllm}}
```

几条经验：

- **污点 + 容忍用来"圈地"**：GPU 节点打上 `nvidia.com/gpu:NoSchedule` 的污点，只有显式容忍的 Pod 才能进来，避免监控、日志这类 Pod 占着 GPU 节点的 CPU。
- **反亲和保证可用性**：同一个模型服务的副本尽量分散，避免一个节点挂掉带走一半容量。注意用 `preferred`（软）而不是 `required`（硬），否则节点不够时 Pod 会直接 Pending。
- **`IgnoredDuringExecution` 的含义**：这些约束只在调度那一刻检查，Pod 跑起来之后节点标签变了也不会被赶走。

## 多机多卡：gang 调度与队列

一个 TP=8、PP=2 的推理实例要 16 张卡、跨两个节点，所有成员必须同时就位才能建立通信组。默认调度器逐个 Pod 调度，没有"整组"的概念，于是会出现：任务 A 抢到 8 张、任务 B 抢到 8 张，两个都差一半，谁也起不来，卡还都占着。

解决方案是 **gang scheduling**（也叫 all-or-nothing）：

| 方案 | 思路 | 适合 |
| --- | --- | --- |
| **Volcano** | 自带调度器，PodGroup 声明 `minMember`，凑齐才绑定；还有队列、公平共享、抢占 | 训练与批处理为主的集群 |
| **Kueue** | 不替换调度器，在"准入"层排队：Workload 攒够资源配额才放行给默认调度器 | 想保留默认调度器、按队列管配额 |
| **YuniKorn** | 替换调度器，强调多租户队列与层级配额 | 混合负载的大集群 |
| **LeaderWorkerSet** | 不做 gang，但把"一个 leader + N 个 worker"打包成一个副本单位（第五章展开） | 多机多卡的**推理**实例 |

推理和训练的侧重点不同：训练任务排队等资源是常态，gang + 队列 + 抢占是刚需；推理服务通常常驻，更关心"扩容时能不能一次拿到一整组卡"以及"滚动更新时新旧实例的资源重叠"。所以推理集群里常见的组合是：LeaderWorkerSet 管副本组，Kueue 管配额与准入，默认调度器加装箱打分。

## 抢占与优先级

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata: {name: inference-critical}
value: 1000000
preemptionPolicy: PreemptLowerPriority
globalDefault: false
description: "在线推理，可抢占离线任务"
```

Pod 里写 `priorityClassName: inference-critical` 之后，它 Pending 时调度器会尝试**抢占**：挑一组低优先级 Pod 驱逐，腾出资源给它。要点：

- 抢占是**按节点**做的：必须在某一个节点上驱逐足够多的 Pod 才能放下候选 Pod，所以碎片化严重时抢占也救不了（就是上面那句 `No preemption victims found`）。
- 被抢占的 Pod 会收到优雅退出信号，`terminationGracePeriodSeconds` 照常生效——推理任务要把这段时间用来做完手上的请求。
- `PodDisruptionBudget` 对抢占**不是硬约束**（调度器会尽量尊重，但优先级高的抢占仍可能突破）。
- 在线推理设高优先级、离线评测和压测设低优先级，是 GPU 集群提高利用率的常用做法：白天在线占着，夜里离线把空卡填满，白天再被抢回去。

!!! interview "怎么讲清楚"
    讲调度：先给两段式——过滤（资源、亲和、污点，淘汰不可行节点）和打分（默认 LeastAllocated 摊开，GPU 集群常改成 MostAllocated 装箱以空出整机）。然后讲 GPU 的特殊性：不可压缩、只能整数、requests 必须等于 limits、由 device plugin 上报，因此碎片化是常态，缓解靠装箱、按卡数分池、预留和重调度。多机多卡要 gang 调度，否则多个任务各抢一半互相卡死，方案有 Volcano（PodGroup minMember）、Kueue（准入层排队）、LWS（副本组）。再补优先级与抢占：在线推理高优先级抢占离线任务，但抢占是按节点做的，碎片化时也救不了。最后给排障句式：看 `FailedScheduling` 那行，它直接写明每个节点被淘汰的原因。

## 练习

**1. 读一条调度失败信息。** 事件写着 `0/12 nodes are available: 3 Insufficient nvidia.com/gpu, 8 node(s) had untolerated taint {workload: training}, 1 node(s) didn't match Pod's node affinity`。这个集群发生了什么？你会怎么改？

??? success "参考答案"
    12 个节点全被淘汰：3 个 GPU 不够、8 个有训练专用污点而这个 Pod 没容忍、1 个节点亲和性不匹配。说明这个 Pod 本来就只能落在那 3 个 GPU 节点上，而它们的卡被占满了。

    三条路：（1）等或扩容——这是资源问题，不是配置问题；（2）如果这个推理任务本来就允许跑在训练节点上，给它加 `tolerations`，并配上优先级让它能抢占离线任务；（3）检查那 3 个节点上是否有可以驱逐的低优先级 Pod，给推理服务设 `priorityClassName`。

**2. 选打分策略。** 一个 GPU 集群同时跑：很多单卡在线推理、少量 8 卡的大模型实例。用默认的 `LeastAllocated` 会发生什么？改成装箱有什么代价？

??? success "参考答案"
    `LeastAllocated` 会把单卡任务均匀撒到所有节点上，很快就没有任何节点能凑齐 8 张卡——大实例永远 Pending。改成装箱（`MostAllocated` 或 `RequestedToCapacityRatio`）能把单卡任务挤在少数节点上，空出整机。

    代价：单点故障影响面变大（一个节点挂掉带走更多副本）、热点节点的 CPU/网络竞争更激烈。折中做法是分池：把集群按"单卡池"和"整机池"分开，用污点和标签隔离，两边各用合适的策略；或者给大实例预留节点。

**3. 估算碎片。** 10 个 8 卡节点，随机放入 40 个单卡任务（平均每节点 4 张）。集群还剩 40 张卡，此时最多能放下几个 8 卡任务？如果单卡任务用装箱策略放呢？

??? success "参考答案"
    随机（或摊开）放置时，每个节点大约剩 4 张，没有一个节点能放下 8 卡任务——**一个都放不下**，尽管还剩一半的卡。

    装箱时，40 个单卡任务会填满 5 个节点，剩下 5 个节点完全空闲，能放下 5 个 8 卡任务。差别就是打分策略。实践里还要加上"节点内的卡也要 NVLink 相连"这个约束，碎片问题只会更严重。

**4. 设计一个准入方案。** 你要在一个共享集群上保证：在线推理随时有资源，离线评测只用空闲卡，且离线任务不能让在线任务等待超过 1 分钟。写出用到的 Kubernetes 机制。

??? success "参考答案"
    （1）两个 `PriorityClass`：在线 1000000、离线 100，在线开启抢占；（2）离线任务的 `terminationGracePeriodSeconds` 设小（比如 30 秒）并实现幂等的检查点，保证被抢占时能快速让出且不丢结果；（3）用 Kueue 或 Volcano 给离线任务设配额队列，限制它最多占多少卡，避免抢占风暴；（4）在线服务配 `PodDisruptionBudget` 防止运维操作一次拿走太多副本；（5）监控上盯住"在线 Pod 从 Pending 到 Running 的时间"，超过阈值就说明抢占链路有问题。

## 小结

- [x] 调度 = 过滤（给出每个节点被淘汰的原因）+ 打分（默认摊开，GPU 集群常改装箱）+ 绑定。
- [x] GPU 不可压缩、只能整数、requests 必须等于 limits、由 device plugin 上报。
- [x] 碎片化让"总量够"和"放不下"同时成立；缓解靠装箱、分池、预留、重调度。
- [x] 多机多卡需要 gang 调度（Volcano 的 PodGroup、Kueue 的准入队列），否则多个任务各抢一半互相卡死。
- [x] 优先级 + 抢占让在线推理压住离线任务，但抢占按节点进行，碎片化时也无能为力。
- [x] Pending 先看 `FailedScheduling` 那行，它把每类原因的节点数都写清楚了。
