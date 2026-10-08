# The scheduler: from filter/score to GPU and gang scheduling

<p class="lead">"Why is my inference Pod stuck in Pending" is the question platform roles get asked most, and the answer is almost always in the scheduler: not enough resources, mismatched labels, an untolerated taint, or machines so fragmented that an 8-GPU task won't fit. This chapter first writes the default scheduler's two-phase flow (filter + score) as a runnable mini scheduler, then covers three things specific to inference workloads (GPUs are incompressible integer resources, multi-machine multi-GPU needs gang scheduling, and fragmentation lets "plenty of GPUs left" and "the task won't fit" both be true), and finally gives a few ways to extend the scheduler.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What two steps does the default scheduler take to pick a node for a Pod? What does each do?
    2. Which scenarios do the `LeastAllocated` and `MostAllocated` scoring strategies each suit?
    3. Why do multi-machine multi-GPU tasks need gang scheduling? What happens without it?
    4. The cluster has 8 GPUs left; why won't a 4-GPU task fit?
    5. For a Pending Pod, which line of `describe` should you read?

??? success "Answers (try first, then expand to compare)"
    1. **Filter (predicate)**: judge node by node "does it fit": enough resources (counted by requests), matching nodeSelector/affinity, tolerated taints, no port conflicts and so on, eliminating any node that fails. **Score**: score the remaining nodes and take the highest; by default it combines plugins for resource balance, image locality, topology spreading and more.
    2. `LeastAllocated` (the default) spreads load out, suited to online services, with a small blast radius for single failures and headroom for bursts; `MostAllocated` is bin packing, filling a few nodes, which frees whole machines for big tasks and helps scaling down to save money, suited to offline tasks and GPU clusters.
    3. A multi-machine multi-GPU task can start only when all members are up (to build its communication group); if one member can't get GPUs, the others hold theirs idle while waiting, and several such tasks blocking each other is a deadlock. Gang scheduling (Volcano, Kueue, YuniKorn) guarantees "schedule all or none".
    4. Because the GPUs are scattered across nodes: say three nodes have 2, 2 and 4 left, 8 in total, but no node can give 4 at once (assuming the task needs them on one machine). This is fragmentation, mitigated by bin-packing scoring, pooling by GPU count, and periodic consolidation (rescheduling).
    5. `FailedScheduling` in `Events`, which spells out why each node was eliminated, for example `0/1 nodes are available: 1 Insufficient nvidia.com/gpu`.

## Two phases: filter first, then score {#两段式先过滤再打分}

![Figure: kube-scheduler's two phases: filter, score, bind](../assets/figures/k8s-filter-score.svg){.aig-svg}

kube-scheduler's core loop: take an unscheduled Pod from the queue → filter the feasible nodes → score the feasible nodes → pick the highest score → bind (write `pod.spec.nodeName`). Written out:

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
    if node.taints - set(pod.get("tolerations", ())):            # taints and tolerations
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
            for n, p in placed:                                  # roll back the ones already placed
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
for i in range(12):                                   # first scatter the cluster with single-GPU tasks
    schedule(nodes, {"name": f"s{i}", "cpu": 4, "mem": 16, "gpu": 1})
print("  当前每个节点剩余 GPU：", {n.name: n.free()[2] for n in nodes if n.gpus})
small_gang = [{"name": f"train-{i}", "cpu": 8, "mem": 32, "gpu": 2} for i in range(2)]
print("  两个 2 卡的成员：", schedule_gang(nodes, small_gang))
for n in nodes:                                       # undo the last attempt and try a bigger gang
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

```text title="output"
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

Read the output for four things:

- **Filtering reports "why not"**, exactly the content of the `FailedScheduling` line in `kubectl describe pod`.
- **The scoring strategy decides the shape of the distribution**: the same 6 single-GPU requests spread into 3 + 3 under `LeastAllocated` and pack into 6 + 0 under `MostAllocated`. GPU clusters usually want the latter: only freed-up whole machines can take 8-GPU tasks.
- **Gang scheduling gives all or nothing**: when the two 3-GPU members don't fit, the whole group rolls back, leaving no half-done state of "holding 3 GPUs while waiting for the other half".
- **Fragmentation is "enough in total but doesn't fit"**: 8 GPUs are left, but only 4 of them can form a complete 4-GPU task.

The real scheduler's plugin framework has more extension points (`PreFilter`, `Filter`, `PostFilter`, `PreScore`, `Score`, `Reserve`, `Permit`, `Bind`), and common plugins include `NodeResourcesFit` (resources), `NodeAffinity` (affinity), `TaintToleration` (taints), `PodTopologySpread` (spreading across zones/racks), `InterPodAffinity` (staying near or away from certain Pods) and `ImageLocality` (the image is already local).

## Why GPUs are special {#gpu-为什么特殊}

| | CPU / memory | GPU (`nvidia.com/gpu`) |
| --- | --- | --- |
| Compressible | CPU is (throttled when exceeded) | no |
| Fractional | CPU is (`500m`) | no, whole numbers only |
| requests / limits | may differ | must be equal (writing limits is enough) |
| Overcommit | common | not allowed by default |
| Reported by | the kubelet directly | the **device plugin** (expanded in the next chapter) |

Three consequences follow:

1. **One GPU can go to only one container** (unless split with MIG or shared by time slicing; see the next chapter). So a need like "half a GPU for a small model" cannot be expressed in Kubernetes' native model.
2. **Fragmentation is the norm in GPU clusters**. An 8-GPU node with 3 GPUs taken by 3 single-GPU tasks has 5 left and can't take an 8-GPU task. Mitigations: bin-packing scores, pools by GPU count (1-GPU pool / 2-GPU pool / whole-machine pool), reserving nodes for big tasks (`node taint` + a dedicated queue), and periodic rescheduling (descheduler) to move scattered tasks together.
3. **Topology matters more than count**. The 8 GPUs on one node also divide into NVLink domains and PCIe topology, and a multi-GPU task needs GPUs that are "directly connected to each other by NVLink", or communication is an order of magnitude slower. This takes NVIDIA's GPU Operator plus a topology-aware allocation policy (or DRA; see the next chapter).

Requesting a GPU on a cluster without GPUs produces this event:

```bash
kubectl apply -f pending.yaml    # the Pod has limits: {nvidia.com/gpu: 1}
kubectl describe pod needs-gpu | sed -n '/Events/,$p'
```

```text title="output (on this machine)"
Events:
  Type     Reason            Age   From               Message
  ----     ------            ----  ----               -------
  Warning  FailedScheduling  6s    default-scheduler  0/1 nodes are available: 1 Insufficient nvidia.com/gpu. preemption: 0/1 nodes are available: 1 No preemption victims found for incoming pod.
```

This line packs in a lot: **how many of how many nodes are available**, **why each was eliminated**, and **whether preemption can help** (`No preemption victims found` means even preempting lower-priority Pods can't free the resources).

## Steering Pods to the right nodes {#把-pod-引导到正确的节点}

```yaml
spec:
  nodeSelector:                       # simplest: labels that must match
    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
  tolerations:                        # tolerate the GPU nodes' taint (GPU nodes are usually tainted to keep ordinary Pods out)
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  affinity:
    nodeAffinity:                     # more flexible: hard constraints + soft preferences
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - {key: node.kubernetes.io/instance-type, operator: In, values: [p5.48xlarge, p4d.24xlarge]}
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
              - {key: topology.kubernetes.io/zone, operator: In, values: [cn-north-1a]}
    podAntiAffinity:                  # spread replicas of the same service across nodes where possible
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          podAffinityTerm:
            labelSelector: {matchLabels: {app: vllm}}
            topologyKey: kubernetes.io/hostname
  topologySpreadConstraints:          # finer spreading: replica counts across zones differ by at most 1
    - maxSkew: 1
      topologyKey: topology.kubernetes.io/zone
      whenUnsatisfiable: ScheduleAnyway
      labelSelector: {matchLabels: {app: vllm}}
```

A few lessons:

- **Taints + tolerations "fence off" nodes**: taint GPU nodes with `nvidia.com/gpu:NoSchedule` so only Pods that explicitly tolerate it can get in, keeping monitoring and logging Pods from taking GPU nodes' CPU.
- **Anti-affinity ensures availability**: spread replicas of the same model service as much as possible, so one failed node doesn't take half the capacity with it. Use `preferred` (soft) rather than `required` (hard), or Pods go straight to Pending when nodes run short.
- **What `IgnoredDuringExecution` means**: these constraints are checked only at scheduling time; a running Pod is not evicted if the node's labels change later.

## Multi-machine multi-GPU: gang scheduling and queues {#多机多卡gang-调度与队列}

An inference instance with TP=8 and PP=2 needs 16 GPUs across two nodes, and all members must be in place at once to build the communication group. The default scheduler schedules Pod by Pod with no notion of a "whole group", so this can happen: task A grabs 8 GPUs, task B grabs 8, both are half short, neither can start, and the GPUs stay occupied.

The solution is **gang scheduling** (also called all-or-nothing):

| Option | Idea | Suits |
| --- | --- | --- |
| **Volcano** | its own scheduler; a PodGroup declares `minMember` and binds only when complete; also has queues, fair sharing and preemption | clusters mostly running training and batch jobs |
| **Kueue** | doesn't replace the scheduler; queues at the "admission" layer: a Workload is released to the default scheduler only once its resource quota is secured | keeping the default scheduler while managing quotas by queue |
| **YuniKorn** | replaces the scheduler, emphasizing multi-tenant queues and hierarchical quotas | large clusters with mixed workloads |
| **LeaderWorkerSet** | not gang scheduling, but packages "one leader + N workers" as one replica unit (expanded in chapter five) | multi-machine multi-GPU **inference** instances |

Inference and training emphasize different things: training tasks queueing for resources is normal, so gang + queues + preemption are essential; inference services are usually long-running and care more about "getting a whole group of GPUs at once when scaling up" and "the resource overlap of old and new instances during rolling updates". So a common combination in inference clusters is: LeaderWorkerSet for replica groups, Kueue for quotas and admission, and the default scheduler with bin-packing scores.

## Preemption and priority {#抢占与优先级}

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata: {name: inference-critical}
value: 1000000
preemptionPolicy: PreemptLowerPriority
globalDefault: false
description: "在线推理，可抢占离线任务"
```

Once a Pod has `priorityClassName: inference-critical`, the scheduler tries **preemption** while it is Pending: pick a set of lower-priority Pods to evict and free resources for it. Key points:

- Preemption works **per node**: enough Pods must be evicted on a single node for the candidate Pod to fit, so under severe fragmentation preemption can't help either (that's the `No preemption victims found` above).
- Preempted Pods get a graceful shutdown signal, and `terminationGracePeriodSeconds` applies as usual; inference tasks should use this time to finish the requests in hand.
- `PodDisruptionBudget` is **not a hard constraint** on preemption (the scheduler tries to respect it, but higher-priority preemption can still break through).
- High priority for online inference and low priority for offline evaluation and load tests is a common way to raise GPU cluster utilization: online holds the GPUs by day, offline fills the idle ones at night, and gets preempted back by day.

!!! interview "How to explain it"
    To explain scheduling: give the two phases first: filter (resources, affinity, taints, eliminating infeasible nodes) and score (LeastAllocated spreads by default; GPU clusters often switch to MostAllocated bin packing to free whole machines). Then GPUs' special nature: incompressible, whole numbers only, requests must equal limits, reported by the device plugin, so fragmentation is the norm, mitigated by bin packing, pooling by GPU count, reservation and rescheduling. Multi-machine multi-GPU needs gang scheduling, or several tasks each grab half and deadlock; options are Volcano (PodGroup minMember), Kueue (queueing at admission), and LWS (replica groups). Add priority and preemption: online inference at high priority preempts offline tasks, but preemption works per node and can't help under fragmentation. Finally, the troubleshooting line: read the `FailedScheduling` line, which states plainly why each node was eliminated.

## Exercises {#练习}

**1. Read a scheduling failure.** The event says `0/12 nodes are available: 3 Insufficient nvidia.com/gpu, 8 node(s) had untolerated taint {workload: training}, 1 node(s) didn't match Pod's node affinity`. What happened in this cluster? How would you fix it?

??? success "Answer"
    All 12 nodes were eliminated: 3 lack GPUs, 8 have a training-only taint this Pod doesn't tolerate, and 1 fails node affinity. So this Pod could only ever land on those 3 GPU nodes, and their GPUs are all taken.

    Three paths: (1) wait or scale up; this is a resource problem, not a configuration problem; (2) if this inference task is allowed to run on training nodes, add `tolerations`, together with a priority so it can preempt offline tasks; (3) check whether those 3 nodes have lower-priority Pods that can be evicted, and give the inference service a `priorityClassName`.

**2. Choose a scoring strategy.** A GPU cluster runs both many single-GPU online inference services and a few 8-GPU large-model instances. What happens with the default `LeastAllocated`? What does switching to bin packing cost?

??? success "Answer"
    `LeastAllocated` scatters single-GPU tasks evenly across all nodes, and soon no node can gather 8 GPUs: the large instances stay Pending forever. Switching to bin packing (`MostAllocated` or `RequestedToCapacityRatio`) squeezes single-GPU tasks onto a few nodes and frees whole machines.

    The costs: a larger blast radius for single failures (one failed node takes more replicas with it), and fiercer CPU/network competition on hot nodes. A compromise is pooling: split the cluster into a "single-GPU pool" and a "whole-machine pool", isolated with taints and labels, each with a suitable strategy; or reserve nodes for large instances.

**3. Estimate fragmentation.** 10 nodes with 8 GPUs each, with 40 single-GPU tasks placed at random (4 per node on average). The cluster has 40 GPUs left; how many 8-GPU tasks fit at most? And if the single-GPU tasks were placed with bin packing?

??? success "Answer"
    With random (or spread) placement, each node has about 4 left, and no node fits an 8-GPU task: **not a single one fits**, even though half the GPUs are left.

    With bin packing, the 40 single-GPU tasks fill 5 nodes, leaving 5 nodes completely idle, enough for 5 8-GPU tasks. The difference is the scoring strategy. In practice, add the constraint that "GPUs within a node must also be NVLink-connected", and fragmentation only gets worse.

**4. Design an admission scheme.** On a shared cluster you must guarantee that online inference always has resources, offline evaluation uses only idle GPUs, and offline tasks never make online tasks wait more than 1 minute. Name the Kubernetes mechanisms you'd use.

??? success "Answer"
    (1) Two `PriorityClass`es: online at 1000000 and offline at 100, with preemption enabled for online; (2) a small `terminationGracePeriodSeconds` for offline tasks (say 30 seconds) plus idempotent checkpoints, so preempted tasks yield quickly without losing results; (3) quota queues for offline tasks with Kueue or Volcano, capping how many GPUs they can hold and avoiding preemption storms; (4) a `PodDisruptionBudget` for online services so maintenance never takes too many replicas at once; (5) monitoring of "the time for online Pods to go from Pending to Running", where exceeding a threshold means something is wrong in the preemption path.

## Summary {#小结}

- [x] Scheduling = filter (reporting why each node was eliminated) + score (spread by default, often switched to bin packing in GPU clusters) + bind.
- [x] GPUs are incompressible, whole numbers only, with requests equal to limits, reported by the device plugin.
- [x] Fragmentation makes "enough in total" and "won't fit" true at once; mitigate with bin packing, pooling, reservation and rescheduling.
- [x] Multi-machine multi-GPU needs gang scheduling (Volcano's PodGroup, Kueue's admission queues), or several tasks each grab half and deadlock.
- [x] Priority + preemption let online inference override offline tasks, but preemption works per node and is powerless under fragmentation.
- [x] For Pending, read the `FailedScheduling` line first; it states how many nodes failed for each reason.
