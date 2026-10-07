# How GPUs are managed in Kubernetes

<p class="lead">Behind the line `resources.limits."nvidia.com/gpu": 1` is a chain from the driver and the container runtime through the device plugin to the scheduler. If any link in the chain is misconfigured, the Pod gets stuck in Pending or starts without seeing its GPU. This chapter explains the chain, then does three sets of accounting: how much slower communication gets when topology picks the wrong GPUs, how much each slice keeps after MIG splits a GPU, and what time-slice sharing costs in latency; finally it gives criteria for "when to dedicate a GPU and when to split it".</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Between "`nvidia-smi` shows the GPU" and "the Pod can use the GPU", which components are involved?
    2. What does the device plugin do? Why isn't the GPU reported directly by the kubelet like CPU is?
    3. Among 8 GPUs in one machine, how different is picking any 4 versus picking 4 "in the same NVLink group"?
    4. How do MIG and time-slice sharing differ? What does each cost?
    5. Why taint GPU nodes?

??? success "Answers (try first, then expand to compare)"
    1. The host's NVIDIA driver → GPU support in the container runtime (nvidia-container-toolkit, letting containers see the device nodes and driver libraries) → the **device plugin** (reports "how many GPUs this machine has" to the kubelet and injects devices into containers at allocation) → the scheduler filtering on the extended resource `nvidia.com/gpu`. In production these are usually installed all at once by the **GPU Operator**, plus DCGM Exporter for metrics and MIG Manager for splitting.
    2. Because GPUs are **extended resources**: the kubelet knows only a few built-in resources (CPU, memory, ephemeral storage), and every other device is reported through the device plugin's gRPC interface (`ListAndWatch` reports the device list and health, `Allocate` returns the device nodes, environment variables and mounts to inject into the container at allocation). This way vendors don't have to change Kubernetes itself.
    3. In this chapter's model, an 8-GPU machine has two groups with NVLink direct connections within each: the 4 GPUs picked first-come-first-served by number span both groups, at a communication cost of 18; topology-aware picking gets 4 from the same group at a cost of 6, a 3× difference. In real systems, crossing NVLink domains goes over PCIe with an order of magnitude less bandwidth, and TP's all-reduce becomes the bottleneck.
    4. **MIG** splits a GPU in hardware into isolated instances (SMs, L2, memory channels and memory all separate), with predictable performance and good fault isolation, but fixed profiles, a GPU reset to re-split, and less memory per slice; **time slicing** has several processes take turns on the whole GPU, flexible but without isolation: memory must be divided by hand, they jitter each other, and one crashing process can drag others down. This chapter's accounting: an H100 running a 7B model with 14 GB of weights takes 4.2 ms per step when dedicated, 14.6 ms per step on a MIG slice of a 3-way split (but serving 3 instances), and 12.5 ms per step with 3 time-sliced tenants that interfere with each other.
    5. To keep Pods that don't need GPUs (monitoring, logging, ordinary services) from taking the GPU nodes' CPU and memory, reserving the expensive machines for workloads that really need GPUs. The way is `kubectl taint nodes <node> nvidia.com/gpu=present:NoSchedule`, with matching `tolerations` in inference Pods.

## What a GPU passes through to reach a container {#一张卡到达容器要经过什么}

![Figure: how a GPU reaches a container: driver, device plugin, scheduler, container runtime](../assets/figures/gpu-to-pod.svg){.aig-svg}

<!-- i18n:diagram 09e14e7c61 -->
```text
host
├── NVIDIA driver (kernel module + /dev/nvidia*)
├── containerd / CRI-O
│   └── nvidia-container-toolkit: injects device nodes, driver libraries and env vars at container start
├── kubelet
│   └── device plugin (DaemonSet)
│        ├── ListAndWatch: reports "nvidia.com/gpu: 8" and each GPU's health
│        └── Allocate: on allocation, returns the devices and env vars to inject (NVIDIA_VISIBLE_DEVICES)
└── kube-scheduler: filters on nvidia.com/gpu as an ordinary extended resource
```

Mapped to operational commands:

```bash
kubectl get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
kubectl -n gpu-operator get pods                    # are the device plugin, DCGM and MIG manager all Running
kubectl describe node <gpu-node> | grep -A5 Allocated   # how many GPUs are allocated
kubectl exec -it <pod> -- nvidia-smi                # how many GPUs the container can see
```

The **GPU Operator** packages the whole set (driver, toolkit, device plugin, DCGM Exporter, MIG Manager, node feature discovery) as a group of DaemonSets and CRDs that install automatically once a node is labeled, and it also adds labels like `nvidia.com/gpu.product` and `nvidia.com/gpu.memory` to nodes for scheduling.

Common failures:

| Symptom | Cause |
| --- | --- |
| No `nvidia.com/gpu` in the node's `allocatable` | the device plugin isn't up, or the driver isn't installed properly (check the DaemonSet logs) |
| Pod Pending with `Insufficient nvidia.com/gpu` | GPUs all taken, or node labels/taints don't match |
| Pod Running but `nvidia-smi` errors inside the container | the container runtime isn't configured with the nvidia runtime, or the image lacks the driver libraries |
| A GPU "disappeared" | the device plugin reported an unhealthy device (XID errors, ECC faults); check node events and `dmesg` |

## Which GPUs to pick: topology matters {#选哪几张卡拓扑很重要}

The 8 GPUs in one machine are not equivalent: they belong to different NVLink groups and PCIe switches. If a multi-GPU task gets GPUs across groups, TP's all-reduce takes the slow links (see [CS Fundamentals: multi-GPU systems](root://cs/arch/multi-gpu/)).

```python title="gpualloc.py"
"""GPU 分配：拓扑感知的选卡、MIG 切分与时间片共享的账。

一台 8 卡机的 NVLink 拓扑（简化成"两组四卡"的常见形态）：组内直连，跨组要绕。
真实拓扑用 nvidia-smi topo -m 看，Kubernetes 侧由 device plugin 的拓扑感知分配或 DRA 处理。
"""
from itertools import combinations

GROUPS = [(0, 1, 2, 3), (4, 5, 6, 7)]            # GPUs in the same group are directly connected by NVLink


def link_cost(gpus):
    """一组卡之间的通信代价：同组算 1，跨组算 4（数值只表示量级）"""
    cost = 0
    for a, b in combinations(sorted(gpus), 2):
        same = any(a in g and b in g for g in GROUPS)
        cost += 1 if same else 4
    return cost


def pick_gpus(free, n, topology_aware=True):
    """从空闲卡里挑 n 张：拓扑感知时选通信代价最小的一组，否则按编号先到先得"""
    free = sorted(free)
    if len(free) < n:
        return None
    if not topology_aware:
        return free[:n]
    return min(combinations(free, n), key=lambda c: (link_cost(c), c))


MIG_PROFILES = {                                  # some MIG profiles of an H100 80GB: (slices, memory GB, SM share)
    "1g.10gb": (1, 10, 1 / 7),
    "2g.20gb": (2, 20, 2 / 7),
    "3g.40gb": (3, 40, 3 / 7),
    "7g.80gb": (7, 80, 1.0),
}


def mig_layout(profile, card_mem_gb=80):
    """一张卡切成这个档位能得到几个实例、每个实例多少显存和多少算力比例"""
    slices, mem, share = MIG_PROFILES[profile]
    return 7 // slices, mem, share


def decode_ms(weight_gb, bandwidth_gbs, share=1.0):
    """batch=1 的 decode 下限：读一遍权重。MIG 按份额切带宽，时间片共享则是排队"""
    return weight_gb / (bandwidth_gbs * share) * 1000


def timeslice_latency(base_ms, tenants):
    """时间片共享：n 个租户轮流用整张卡，单个请求的等待时间线性变差（显存还互相挤）"""
    return base_ms * tenants
```

```python
from gpualloc import decode_ms, link_cost, mig_layout, pick_gpus, timeslice_latency

print("一台 8 卡机：0-3 一组、4-7 一组，同组 NVLink 直连")
free = [2, 3, 4, 5, 6, 7]
for n in (2, 4):
    naive = pick_gpus(free, n, topology_aware=False)
    smart = pick_gpus(free, n)
    print(f"  要 {n} 张卡，空闲 {free}：按编号选 {list(naive)}（代价 {link_cost(naive)}），"
          f"拓扑感知选 {list(smart)}（代价 {link_cost(smart)}）")

print("\nMIG：一张 H100 切成几份，每份多少资源")
for profile in ("1g.10gb", "2g.20gb", "3g.40gb", "7g.80gb"):
    count, mem, share = mig_layout(profile)
    print(f"  {profile:8s} 一张卡切 {count} 份，每份 {mem:2d} GB 显存、{share:.0%} 算力，"
          f"3 GB 权重的 decode 下限 {decode_ms(3, 3350, share):5.1f} ms")

print("\n共享一张卡的三种方式，跑同一个 7B 模型（14 GB 权重，H100 3350 GB/s）：")
base = decode_ms(14, 3350)
print(f"  独占整卡：       每步 {base:5.2f} ms，1 个实例")
count, mem, share = mig_layout("2g.20gb")
print(f"  MIG 2g.20gb：    每步 {decode_ms(14, 3350, share):5.2f} ms，{count} 个实例，互相硬隔离")
for tenants in (2, 3):
    print(f"  时间片共享 {tenants} 个： 每步 {timeslice_latency(base, tenants):5.2f} ms，"
          f"{tenants} 个实例，显存要自己分，互相有抖动")
print("\n注意 MIG 把显存也切了：14 GB 的权重放不进 2g.20gb 之外的小档位，1g.10gb 只能跑更小的模型。")
```

```text title="输出"
一台 8 卡机：0-3 一组、4-7 一组，同组 NVLink 直连
  要 2 张卡，空闲 [2, 3, 4, 5, 6, 7]：按编号选 [2, 3]（代价 1），拓扑感知选 [2, 3]（代价 1）
  要 4 张卡，空闲 [2, 3, 4, 5, 6, 7]：按编号选 [2, 3, 4, 5]（代价 18），拓扑感知选 [4, 5, 6, 7]（代价 6）

MIG：一张 H100 切成几份，每份多少资源
  1g.10gb  一张卡切 7 份，每份 10 GB 显存、14% 算力，3 GB 权重的 decode 下限   6.3 ms
  2g.20gb  一张卡切 3 份，每份 20 GB 显存、29% 算力，3 GB 权重的 decode 下限   3.1 ms
  3g.40gb  一张卡切 2 份，每份 40 GB 显存、43% 算力，3 GB 权重的 decode 下限   2.1 ms
  7g.80gb  一张卡切 1 份，每份 80 GB 显存、100% 算力，3 GB 权重的 decode 下限   0.9 ms

共享一张卡的三种方式，跑同一个 7B 模型（14 GB 权重，H100 3350 GB/s）：
  独占整卡：       每步  4.18 ms，1 个实例
  MIG 2g.20gb：    每步 14.63 ms，3 个实例，互相硬隔离
  时间片共享 2 个： 每步  8.36 ms，2 个实例，显存要自己分，互相有抖动
  时间片共享 3 个： 每步 12.54 ms，3 个实例，显存要自己分，互相有抖动

注意 MIG 把显存也切了：14 GB 的权重放不进 2g.20gb 之外的小档位，1g.10gb 只能跑更小的模型。
```

The first section shows the value of topology awareness: with free GPUs `[2,3,4,5,6,7]`, picking 4 by number spans both groups (cost 18), while topology-aware picking gets `[4,5,6,7]` from one group (cost 6).

How to get Kubernetes to do this:

- **Topology-aware allocation in the device plugin**: the NVIDIA device plugin has an allocation policy that picks GPUs by NVLink affinity, used with the topology labels applied by GPU Feature Discovery;
- **The kubelet's Topology Manager**: `--topology-manager-policy=single-numa-node` keeps GPUs, NICs and CPU cores on the same NUMA node where possible, avoiding cross-socket traffic (see [CS Fundamentals: pinned memory and NUMA](root://cs/os/pinned-numa/));
- **DRA (Dynamic Resource Allocation)**: a new mechanism in beta since 1.32 that uses `ResourceClaim` to express structured needs like "I want 4 GPUs directly connected to each other by NVLink", far more precise than "I want 4 nvidia.com/gpu"; it is where things are heading.

## Splitting a GPU: MIG and time slicing {#切分一张卡mig-与时间片}

One H100 running a 0.6B small model is a waste, but Kubernetes GPU resources are whole numbers only. Three paths:

| Approach | Isolation | Memory | Suits |
| --- | --- | --- | --- |
| **MIG** | hardware-level (SMs, L2 and memory channels all separate) | fixed per slice, split by profile | multi-tenancy needing predictable performance; fixed profiles, and re-splitting needs a GPU reset |
| **Time slicing** | none, relies on driver rotation | shared, must be constrained by hand | development and testing, small low-load models; jittery |
| **MPS** | none, but kernels can truly run concurrently | shared | multiple processes of the same tenant, more efficient than time slicing |

The second and third sections of the program above do the accounting: after an H100 is split into `2g.20gb` (3 slices), each slice has only about 29% of the bandwidth, and the same 7B model's decode goes from 4.2 ms to 14.6 ms, but **serves 3 instances at once**, so overall throughput rises; with 3 time-sliced tenants each step takes 12.5 ms, with jitter between them and memory to be divided by hand.

What the Kubernetes-side configuration looks like:

```yaml
# time slicing: have the device plugin "over-report" one GPU as 4
apiVersion: v1
kind: ConfigMap
metadata: {name: device-plugin-config, namespace: gpu-operator}
data:
  any: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4          # the node reports 4× the GPUs, and 4 Pods take turns on the same GPU
```

```yaml
# MIG: label the node, MIG Manager splits GPUs by profile and reports them under a new resource name
metadata:
  labels:
    nvidia.com/mig.config: all-1g.10gb
---
resources:
  limits:
    nvidia.com/mig-1g.10gb: 1    # the Pod requests one MIG instance
```

The selection criteria are simple: **do the model weights fit in a slice's memory** (MIG's memory per slice is a hard cap), **is the service sensitive to latency jitter** (if so, avoid time slicing), and **is it multi-tenant** (crossing tenants requires hard isolation). Inference services usually either dedicate a whole GPU or use MIG for multi-tenant small models; time slicing is used more in development environments.

## Keeping GPU nodes for what belongs there {#让-gpu-节点只跑该跑的东西}

```bash
kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule
kubectl label nodes gpu-node-1 node-role.kubernetes.io/gpu=true
```

```yaml
spec:
  tolerations:
    - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
  nodeSelector:
    node-role.kubernetes.io/gpu: "true"
```

Supporting work:

- **`/dev/shm` must be large enough**: most inference frameworks use shared memory for inter-process communication, and containers default to only 64 MB. Mount `emptyDir: {medium: Memory, sizeLimit: 8Gi}` at `/dev/shm`, or multi-process startup fails (see [CS Fundamentals: inter-process communication](root://cs/os/ipc/)).
- **IPC and RDMA**: multi-machine multi-GPU needs `hostNetwork` or a CNI with RDMA support configured (SR-IOV, RoCE device plugins), plus the `IPC_LOCK` capability for the container to pin memory.
- **Node health checks**: GPUs fail (XID errors, ECC, falling off the bus), so there must be detection that automatically taints the node `NoSchedule` and evicts Pods (NVIDIA's node-problem-detector plugin, or an in-house inspection DaemonSet).
- **Images are huge**: inference images are often over ten GB, taking minutes to pull the first time. Shorten cold starts with image pre-warming (a DaemonSet that pulls ahead of time), a local image registry, or lazy loading (stargz, SOCI).

!!! interview "In an interview"
    When asked "how are GPUs managed on K8s": give the chain first: driver → nvidia-container-toolkit → device plugin (ListAndWatch reports, Allocate injects) → the scheduler filtering it as an extended resource; in production the GPU Operator installs it all at once, with DCGM for metrics. Then three key points: (1) **topology**: GPUs in one machine also split into NVLink groups, and picking wrong makes communication several times slower, solved with topology-aware device plugins, the Topology Manager or the new DRA; (2) **splitting**: MIG is hardware isolation, with memory and compute split by profile and predictable performance, while time slicing and MPS share, with jitter; choose by whether the weights fit, whether jitter matters, and whether it's multi-tenant; (3) **operations**: taint GPU nodes to keep stray Pods out, enlarge `/dev/shm`, pre-warm images to shorten cold starts, and isolate failed GPUs automatically. Finally, be able to recite the table for locating common failures (no GPUs in allocatable = plugin problem, Pending = resources or affinity, nvidia-smi errors inside the container = runtime problem).

## Exercises {#练习}

**1. Can MIG be used?** A 7B model with 14 GB of weights must give each of 5 departments one instance, on H100 80GB GPUs. Can MIG be used? How would you split? If not, what are the alternatives?

??? success "Answer"
    `1g.10gb` won't do (10 GB can't hold 14 GB of weights, plus KV Cache). `2g.20gb` is feasible: 20 GB per slice, 3 slices per GPU, leaving only about 5 GB for KV after the 14 GB of weights, so concurrency is very limited. Giving each of 5 departments a slice needs two GPUs (3 + 2 slices).

    Alternatives: (1) **merge the 5 departments' traffic onto one instance**, distinguishing them with prefix caching and priorities; this almost always saves GPUs, since inference throughput rises with batch size; (2) quantize to FP8 for 7 GB of weights, which `1g.10gb` can barely hold but with too little KV space; (3) if only quotas need isolating rather than performance, cap usage with namespaces + ResourceQuota while still sharing GPUs.

**2. The cost of topology.** A TP=4 instance got 4 GPUs spanning NVLink groups. Each layer does two all-reduces of 1 MB each, over 80 layers. Within a group it's 450 GB/s one way; across groups it goes over PCIe 5.0 at about 60 GB/s. How different is one decode step's communication time?

??? success "Answer"
    A ring all-reduce sends about 2(n-1)/n × the data = 1.5 MB per GPU. Within a group: 1.5 MB / 450 GB/s ≈ 3.3 µs; across groups: 1.5 MB / 60 GB/s ≈ 25 µs (plus higher latency). Two per layer, 80 layers: about 0.53 ms within a group, about 4 ms across groups.

    A decode step is only a few milliseconds to begin with, so cross-group communication more than doubles it. This is why multi-GPU instances must get topologically adjacent GPUs, and why "structured resource requests" like DRA exist.

**3. Design a sharing scheme.** A development environment has 4 A100s and 20 engineers running small models for experiments, with someone occasionally needing a whole GPU for training. How would you configure it?

??? success "Answer"
    Pool them: turn on time slicing on 2 GPUs (`replicas: 4`, reported as 8 virtual GPUs) for daily experiments, and keep 2 GPUs dedicated for training. Separate the two pools with node labels and taints, and add a `LimitRange` to Pods in the experiment pool to set default limits so no one person takes everything.

    Add three more things: (1) give experiment Pods low priority and `activeDeadlineSeconds` for automatic reclamation, so nobody hoards idle GPUs; (2) monitor each Pod's GPU utilization (DCGM metrics) and remind owners of persistently low utilization; (3) tell everyone that time-slice sharing has no memory isolation: one person's OOM can hit others on the same GPU, so set `--gpu-memory-utilization` yourself.

**4. Troubleshooting.** A Pod is Running, but `nvidia-smi` inside the container reports `Failed to initialize NVML: Unknown Error`. What could be the cause?

??? success "Answer"
    (1) The container runtime isn't using the nvidia runtime (the RuntimeClass isn't configured, or containerd's default_runtime is wrong), so device nodes weren't injected; (2) the Pod didn't request a GPU (no `nvidia.com/gpu` in `limits`), so the device plugin injects no devices and the container sees no GPU; (3) the driver version is incompatible with the CUDA version in the container; (4) a known compatibility issue in cgroup v2 environments (the device cgroup gets rewritten after the container starts; NVIDIA has an official configuration option for it).

    Investigation order: first `kubectl describe pod` to confirm the requests really include a GPU, then `ls /dev/nvidia*` to see whether the devices made it into the container, and finally `nvidia-smi` on the host and the device plugin's logs.

## Summary {#小结}

- [x] The chain: driver → nvidia-container-toolkit → device plugin (reporting and injection) → the scheduler's extended resources; packaged by the GPU Operator in production.
- [x] GPUs in one machine are not equivalent: picking the wrong topology makes communication several times slower, solved by topology-aware allocation, the Topology Manager and DRA.
- [x] Three ways to split a GPU: MIG with hard isolation but fixed profiles, time slicing flexible but jittery, MPS for multiple processes of one tenant; choose by whether memory fits, whether jitter matters, and whether it's multi-tenant.
- [x] Taint GPU nodes, enlarge `/dev/shm`, pre-warm images, and isolate failed GPUs automatically.
- [x] Three classes of failure: no GPUs in allocatable (plugin), Pending (resources or affinity), nvidia-smi errors in the container (runtime, or no GPU requested).
