# GPU 在 Kubernetes 里怎么被管起来

<p class="lead"><code>resources.limits."nvidia.com/gpu": 1</code> 这一行背后，是一条从驱动、容器运行时、device plugin 一直到调度器的链路。链路上任何一环没配好，Pod 就会卡在 Pending 或者起来看不到卡。这一章讲清这条链路，再算三笔账：拓扑选错卡通信慢多少、MIG 把一张卡切开之后每份还剩多少、时间片共享的延迟代价是什么——最后给出"什么时候该独占、什么时候该切分"的判断标准。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 从 `nvidia-smi` 能看到卡，到 Pod 里能用上卡，中间要哪几个组件？
    2. device plugin 做了什么？为什么 GPU 不像 CPU 那样由 kubelet 直接上报？
    3. 同一台机器上的 8 张卡，随便挑 4 张和挑"同一 NVLink 组"的 4 张，差别有多大？
    4. MIG 和时间片共享有什么区别？各自的代价是什么？
    5. 为什么 GPU 节点上要打污点？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 宿主机的 NVIDIA 驱动 → 容器运行时的 GPU 支持（nvidia-container-toolkit，让容器能看到设备节点和驱动库）→ **device plugin**（向 kubelet 上报"这台机器有几张卡"并在分配时把设备注入容器）→ 调度器按 `nvidia.com/gpu` 这个扩展资源做过滤。生产上这几样通常由 **GPU Operator** 一次装好，外加 DCGM Exporter 出指标、MIG Manager 管切分。
    2. 因为 GPU 是**扩展资源**：kubelet 只认识 CPU、内存、临时存储这几种内置资源，其他设备一律通过 device plugin 的 gRPC 接口上报（`ListAndWatch` 上报设备列表与健康状态，`Allocate` 在分配时返回要注入容器的设备节点、环境变量和挂载）。这样厂商不用改 Kubernetes 本体。
    3. 本章的模型里，8 卡机分两组、组内 NVLink 直连：按编号先到先得挑到的 4 张卡跨了组，通信代价 18；拓扑感知挑到同一组的 4 张，代价 6，差三倍。真实系统里跨 NVLink 域走 PCIe，带宽差一个数量级，TP 的 all-reduce 会成为瓶颈。
    4. **MIG** 在硬件上把一张卡切成互相隔离的实例（SM、L2、显存通道、显存都分开），性能可预期、故障隔离好，但档位固定、切分要重置卡、且每份显存变小；**时间片共享**是多个进程轮流用整张卡，灵活但没有隔离——显存要自己分、互相有抖动、一个进程崩了可能拖累别人。本章的账：一张 H100 跑 14 GB 的 7B 模型，独占每步 4.2 ms，切成 3 份的 MIG 每步 14.6 ms（但能服务 3 个实例），时间片 3 个租户每步 12.5 ms 且互相干扰。
    5. 防止不需要 GPU 的 Pod（监控、日志、普通服务）占用 GPU 节点的 CPU 和内存，把昂贵的机器留给真正要卡的负载。做法是 `kubectl taint nodes <node> nvidia.com/gpu=present:NoSchedule`，推理 Pod 写对应的 `tolerations`。

## 一张卡到达容器要经过什么

![图：一张卡怎么到达容器——驱动、device plugin、调度器、容器运行时](../assets/figures/gpu-to-pod.svg){.aig-svg}

```text
宿主机
├── NVIDIA 驱动（内核模块 + /dev/nvidia*）
├── containerd / CRI-O
│   └── nvidia-container-toolkit：在容器启动时注入设备节点、驱动库、环境变量
├── kubelet
│   └── device plugin（DaemonSet）
│        ├── ListAndWatch：上报 "nvidia.com/gpu: 8" 与每张卡的健康状态
│        └── Allocate：分配时返回要注入的设备与环境变量（NVIDIA_VISIBLE_DEVICES）
└── kube-scheduler：把 nvidia.com/gpu 当成普通的扩展资源做过滤
```

对应到运维动作：

```bash
kubectl get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
kubectl -n gpu-operator get pods                    # device plugin、DCGM、MIG manager 是否都 Running
kubectl describe node <gpu-node> | grep -A5 Allocated   # 已分配多少卡
kubectl exec -it <pod> -- nvidia-smi                # 容器里看得见几张卡
```

**GPU Operator** 把这一套（驱动、toolkit、device plugin、DCGM Exporter、MIG Manager、节点特性发现）打包成一组 DaemonSet 和 CRD，节点打上标签就自动装好，还会给节点加上 `nvidia.com/gpu.product`、`nvidia.com/gpu.memory` 这类标签供调度使用。

常见故障对应：

| 现象 | 原因 |
| --- | --- |
| 节点 `allocatable` 里没有 `nvidia.com/gpu` | device plugin 没起来，或驱动没装好（看 DaemonSet 日志） |
| Pod Pending，`Insufficient nvidia.com/gpu` | 卡被占满，或节点标签/污点不匹配 |
| Pod Running 但容器里 `nvidia-smi` 报错 | 容器运行时没配 nvidia runtime，或镜像里缺驱动库 |
| 卡"消失"了一张 | device plugin 上报了不健康设备（XID 错误、ECC 故障），看节点事件和 `dmesg` |

## 选哪几张卡：拓扑很重要

同一台机器上的 8 张卡并不是等价的：它们分属不同的 NVLink 组和 PCIe 交换芯片。多卡任务如果拿到跨组的卡，TP 的 all-reduce 就要走慢链路（见[计算机基础：多卡系统](root://cs/arch/multi-gpu/)）。

```python title="gpualloc.py"
"""GPU 分配：拓扑感知的选卡、MIG 切分与时间片共享的账。

一台 8 卡机的 NVLink 拓扑（简化成"两组四卡"的常见形态）：组内直连，跨组要绕。
真实拓扑用 nvidia-smi topo -m 看，Kubernetes 侧由 device plugin 的拓扑感知分配或 DRA 处理。
"""
from itertools import combinations

GROUPS = [(0, 1, 2, 3), (4, 5, 6, 7)]            # 同组内 NVLink 直连


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


MIG_PROFILES = {                                  # H100 80GB 的部分 MIG 档位：(份数, 显存 GB, SM 占比)
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

第一段说明拓扑感知的价值：空闲卡是 `[2,3,4,5,6,7]` 时，按编号挑 4 张会跨组（代价 18），拓扑感知挑到同一组的 `[4,5,6,7]`（代价 6）。

怎么让 Kubernetes 做到这件事：

- **device plugin 的拓扑感知分配**：NVIDIA device plugin 有一套按 NVLink 亲和性挑卡的分配策略，配合 GPU Feature Discovery 打出的拓扑标签使用；
- **kubelet 的 Topology Manager**：`--topology-manager-policy=single-numa-node` 让 GPU、网卡、CPU 核尽量落在同一个 NUMA 节点，避免跨插槽（见[计算机基础：锁页内存与 NUMA](root://cs/os/pinned-numa/)）；
- **DRA（动态资源分配）**：1.32 起进入 beta 的新机制，用 `ResourceClaim` 表达"我要 4 张互相 NVLink 直连的卡"这类结构化需求，比"我要 4 个 nvidia.com/gpu"精确得多，是未来的方向。

## 切分一张卡：MIG 与时间片

一张 H100 跑一个 0.6B 的小模型太浪费，但 Kubernetes 的 GPU 资源只能整数。三条路：

| 方式 | 隔离 | 显存 | 适合 |
| --- | --- | --- | --- |
| **MIG** | 硬件级（SM、L2、显存通道都分开） | 每份固定，按档位切 | 多租户、要求性能可预期；档位固定，切分要重置卡 |
| **时间片（time-slicing）** | 无，靠驱动轮转 | 共享，要自己约束 | 开发测试、低负载的小模型；有抖动 |
| **MPS** | 无，但 kernel 可以真正并发 | 共享 | 同一租户的多个进程，比时间片效率高 |

上面程序的第二、三段给出了这笔账：一张 H100 切成 `2g.20gb`（3 份）之后，每份的带宽只有约 29%，同一个 7B 模型的 decode 从 4.2 ms 变成 14.6 ms，但**同时服务 3 个实例**，整体吞吐是上升的；时间片 3 个租户每步 12.5 ms，而且互相之间有抖动、显存要自己分。

Kubernetes 侧的配置形态：

```yaml
# 时间片：让 device plugin 把一张卡"虚报"成 4 个
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
            replicas: 4          # 节点上报 4 倍的卡数，4 个 Pod 轮流用同一张卡
```

```yaml
# MIG：给节点打标签，MIG Manager 按档位切卡，切完上报成新的资源名
metadata:
  labels:
    nvidia.com/mig.config: all-1g.10gb
---
resources:
  limits:
    nvidia.com/mig-1g.10gb: 1    # Pod 申请一个 MIG 实例
```

选择的判断标准很简单：**模型权重放得进那一份显存吗**（MIG 的每份显存是硬上限）、**这个服务对延迟抖动敏感吗**（敏感就别用时间片）、**是不是多租户**（跨租户必须硬隔离）。推理服务通常要么独占整卡，要么用 MIG 给小模型做多租户；时间片更多用在开发环境。

## 让 GPU 节点只跑该跑的东西

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

配套的几件事：

- **`/dev/shm` 要够大**：多数推理框架用共享内存做进程间通信，容器默认只有 64 MB。用 `emptyDir: {medium: Memory, sizeLimit: 8Gi}` 挂到 `/dev/shm`，否则多进程启动就报错（见[计算机基础：进程间通信](root://cs/os/ipc/)）。
- **IPC 与 RDMA**：多机多卡要开 `hostNetwork` 或配好 CNI 的 RDMA 支持（SR-IOV、RoCE 设备插件），并给容器 `IPC_LOCK` 权限来锁页内存。
- **节点健康检查**：GPU 会掉（XID 错误、ECC、掉卡），要有探测并自动给节点打 `NoSchedule` 污点 + 驱逐 Pod 的机制（NVIDIA 的 node-problem-detector 插件或自研的巡检 DaemonSet）。
- **镜像很大**：推理镜像常常十几 GB，第一次拉取几分钟。用镜像预热（DaemonSet 提前 pull）、本地镜像仓库、或者延迟加载（stargz、SOCI）缩短冷启动。

!!! interview "怎么讲清楚"
    讲"GPU 在 K8s 上怎么管"：先给链路——驱动 → nvidia-container-toolkit → device plugin（ListAndWatch 上报、Allocate 注入）→ 调度器当扩展资源过滤，生产上用 GPU Operator 一次装好，DCGM 出指标。再讲三个要点：（1）**拓扑**：同机的卡也分 NVLink 组，挑错卡通信慢几倍，靠 device plugin 的拓扑感知、Topology Manager 或新的 DRA 解决；（2）**切分**：MIG 是硬件隔离、显存和算力按档位切、性能可预期，时间片和 MPS 是共享、有抖动，选择标准是权重放不放得下、对抖动敏不敏感、是否多租户；（3）**运维**：GPU 节点打污点防止闲杂 Pod 占用，`/dev/shm` 要调大，镜像预热缩短冷启动，掉卡要能自动隔离。最后能说出常见故障的定位表（allocatable 没有卡 = plugin 问题、Pending = 资源或亲和、容器里 nvidia-smi 报错 = 运行时问题）。

## 练习

**1. 判断能不能用 MIG。** 一个 14 GB 权重的 7B 模型要给 5 个部门各提供一个实例，手上是 H100 80GB。能用 MIG 吗？怎么切？不能的话有什么替代方案？

??? success "参考答案"
    不能切到 `1g.10gb`（10 GB 装不下 14 GB 权重，还要留 KV Cache）。可行的是 `2g.20gb`：每份 20 GB，一张卡切 3 份，权重 14 GB 之后只剩约 5 GB 放 KV，并发很有限。要给 5 个部门各一份，需要两张卡（3 + 2 份）。

    替代方案：（1）把 5 个部门的流量**合并到一个实例**上，用前缀缓存和优先级区分——这几乎总是更省卡，因为推理服务的吞吐随 batch 提升；（2）量化到 FP8，权重 7 GB，`1g.10gb` 勉强能放但 KV 空间太小；（3）如果只是隔离配额而不是隔离性能，用命名空间 + ResourceQuota 管住用量，卡还是共享的。

**2. 拓扑的代价。** 一个 TP=4 的实例拿到了跨 NVLink 组的 4 张卡。每层两次 all-reduce、每次 1 MB，80 层。组内单向 450 GB/s、跨组走 PCIe 5.0 约 60 GB/s。decode 一步的通信时间差多少？

??? success "参考答案"
    环形 all-reduce 每张卡发送约 2(n-1)/n × 数据量 = 1.5 MB。组内：1.5 MB / 450 GB/s ≈ 3.3 µs；跨组：1.5 MB / 60 GB/s ≈ 25 µs（还要加上更高的延迟）。每层两次、80 层：组内约 0.53 ms，跨组约 4 ms。

    decode 一步本来只有几毫秒，跨组通信直接让它翻倍以上。这就是为什么多卡实例必须拿到拓扑相邻的卡——也是 DRA 这类"结构化资源请求"存在的意义。

**3. 设计共享方案。** 开发环境有 4 张 A100，20 个工程师要跑小模型做实验，偶尔有人要整卡跑训练。怎么配置？

??? success "参考答案"
    分池：2 张卡开时间片（`replicas: 4`，上报成 8 个虚拟 GPU）给日常实验用，2 张卡保持独占给训练。用节点标签和污点把两个池隔开，实验池的 Pod 加 `LimitRange` 限制默认 limits，避免一个人占满。

    再加三件事：（1）实验 Pod 设低优先级并配 `activeDeadlineSeconds` 自动回收，防止占着不用；（2）监控每个 Pod 的 GPU 利用率（DCGM 指标），长期低利用率的自动提醒；（3）告诉大家时间片共享没有显存隔离——一个人 OOM 可能连累同卡的其他人，要自己设 `--gpu-memory-utilization`。

**4. 排障。** 一个 Pod 状态是 Running，但容器里 `nvidia-smi` 报 `Failed to initialize NVML: Unknown Error`。可能是什么原因？

??? success "参考答案"
    （1）容器运行时没有用 nvidia runtime（RuntimeClass 没配、或者 containerd 的 default_runtime 不对），设备节点没被注入；（2）Pod 没有申请 GPU（`limits` 里没写 `nvidia.com/gpu`），device plugin 就不会注入设备，此时容器里看不到卡；（3）驱动版本与容器内 CUDA 版本不兼容；（4）cgroup v2 环境下的已知兼容问题（容器启动后设备 cgroup 被重写，NVIDIA 官方有对应的配置项）。

    排查顺序：先 `kubectl describe pod` 确认 requests 里确实有 GPU，再 `ls /dev/nvidia*` 看设备有没有进容器，最后在宿主机上 `nvidia-smi` 和看 device plugin 的日志。

## 小结

- [x] 链路：驱动 → nvidia-container-toolkit → device plugin（上报与注入）→ 调度器的扩展资源；生产上用 GPU Operator 打包。
- [x] 同机的卡不等价：拓扑选错通信慢几倍，靠拓扑感知分配、Topology Manager、DRA 解决。
- [x] 切卡三选一：MIG 硬隔离但档位固定、时间片灵活但有抖动、MPS 适合同租户多进程；判断标准是显存放不放得下、抖动敏不敏感、是否多租户。
- [x] GPU 节点要打污点、`/dev/shm` 要调大、镜像要预热、掉卡要能自动隔离。
- [x] 三类故障：allocatable 没卡（plugin）、Pending（资源或亲和）、容器里 nvidia-smi 报错（运行时或没申请 GPU）。
