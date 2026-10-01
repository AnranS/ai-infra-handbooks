# Kubernetes 的核心对象与控制器模式

<p class="lead">推理平台几乎都跑在 Kubernetes 上：一个模型服务是一个 Deployment，一次压测是一个 Job，多机多卡的实例是一组有编号的 Pod，滚动发布、扩缩容、故障自愈全靠控制器。这一章先把"声明式 API + 控制器循环"这个内核讲透——它解释了为什么 `kubectl apply` 之后什么都没发生也是正常的、为什么删掉 Pod 会自己长回来、为什么控制器可以随时重启。然后过一遍推理服务真正会用到的那些对象，最后给出排障时的第一组命令。本章的 YAML 和输出都在一个真实的单节点集群上跑过。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. "声明式"和"命令式"的区别是什么？`kubectl apply` 之后到底发生了什么？
    2. 控制器的 reconcile 为什么必须是幂等的？
    3. Deployment、ReplicaSet、Pod 三者的关系是什么？滚动更新时谁在动？
    4. 一个 Pod 里的多个容器共享什么、不共享什么？initContainer 和 sidecar 各用来做什么？
    5. requests 和 limits 有什么区别？只写 limits 会怎样？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 命令式是"执行这个动作"（`docker run`），声明式是"我要的最终状态长这样"。`kubectl apply` 只是把一个对象写进 etcd，API server 返回 200 就结束了；真正干活的是各个控制器——它们监听对象变化，比较"期望状态"和"实际状态"，一步步把两者拉近。所以 apply 成功不等于服务可用，要看 `kubectl get`/`describe` 里的实际状态。
    2. 因为控制器随时可能重启、事件可能重复送达、也可能漏掉（所以还有定期的全量 resync）。reconcile 只根据"当前的期望和当前的实际"决定下一步动作，不依赖"上次做了什么"，重复执行就不会出错。
    3. Deployment 管理 ReplicaSet，ReplicaSet 管理 Pod。每次改 Pod 模板，Deployment 会新建一个 ReplicaSet，然后**同时调整新旧两个 ReplicaSet 的副本数**：新的加、旧的减，节奏由 `maxSurge` 和 `maxUnavailable` 控制。回滚就是把旧 ReplicaSet 的副本数再加回去。
    4. 同一个 Pod 里的容器共享网络命名空间（同一个 IP，互相用 `localhost` 通信）和可以挂载同一批卷，但**不共享文件系统和进程空间**（默认）。initContainer 在主容器之前按顺序跑完再退出，适合下载权重、做预检；sidecar 与主容器并存，适合日志采集、指标导出、代理。
    5. requests 用于调度（决定 Pod 能被放到哪个节点）和 CPU 的权重分配；limits 是运行时的上限（CPU 超了被节流，内存超了被 OOMKill）。只写 limits 时 Kubernetes 会把 requests 设成和 limits 一样，QoS 变成 Guaranteed——对 GPU 推理服务这通常正是你要的，但对 CPU 会浪费配额。

## 声明式 API：apply 之后什么都没发生才是正常的

Kubernetes 的所有交互都是在操作**对象**：你把"期望状态"写进 API server（背后是 etcd），控制器负责把现实拉向它。

```yaml title="deploy.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
  labels: {app: vllm}
spec:
  replicas: 3                       # 期望状态：我要 3 个副本
  selector:
    matchLabels: {app: vllm}        # 这个 Deployment 管哪些 Pod，靠标签选择器
  template:                         # Pod 模板：每个副本长什么样
    metadata:
      labels: {app: vllm}
    spec:
      terminationGracePeriodSeconds: 60
      containers:
        - name: server
          image: vllm/vllm-openai:v0.11.0
          args: ["--model", "Qwen/Qwen3-0.6B", "--port", "8000"]
          ports: [{containerPort: 8000}]
          resources:
            limits: {nvidia.com/gpu: 1}
```

```bash
kubectl apply -f deploy.yaml
kubectl get deploy,pods -l app=vllm
```

```text title="输出（本机示例）"
deployment.apps/vllm created

NAME                   READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/vllm   0/3     3            0           12s

NAME                        READY   STATUS              RESTARTS   AGE
pod/vllm-6ccd564c7d-chg4r   0/1     ContainerCreating   0          12s
pod/vllm-6ccd564c7d-mnw9g   0/1     ContainerCreating   0          12s
pod/vllm-6ccd564c7d-ssnjp   0/1     ContainerCreating   0          12s
```

`apply` 只花了几毫秒，它做的全部事情就是把对象写进去。之后是一连串控制器接力：Deployment 控制器造出 ReplicaSet，ReplicaSet 控制器造出 3 个 Pod，调度器给每个 Pod 选节点，节点上的 kubelet 拉镜像、起容器。**任何一环卡住，`kubectl get` 里的状态就会停在那里**——所以排障的第一步永远是看实际状态和事件，而不是重新 apply 一遍。

## 控制器模式：reconcile 循环

每个控制器都在做同一件事：

```text
for {
    期望 := 从 API server 读对象的 spec
    实际 := 从 API server 读对象的 status / 关联的子对象
    if 有差异 { 做一步动作把它们拉近 }
    等下一个事件或定时 resync
}
```

把这个循环写出来，只要三十行：

```python title="reconcile.py"
# 控制器模式：读"期望状态"和"实际状态"，算出差异，只做把两者拉近的那一步动作
from collections import deque


class Cluster:
    """极简的 API server：只存对象，不做任何决策"""

    def __init__(self):
        self.deployments = {}                  # 名字 -> {"replicas": 期望副本数, "image": 版本}
        self.pods = {}                          # 名字 -> {"owner": ..., "image": ..., "phase": ...}
        self.events = []
        self.seq = 0

    def create_pod(self, owner, image):
        self.seq += 1
        name = f"{owner}-{self.seq}"
        self.pods[name] = {"owner": owner, "image": image, "phase": "Pending"}
        self.events.append(f"create {name}")
        return name

    def delete_pod(self, name):
        self.pods.pop(name, None)
        self.events.append(f"delete {name}")

    def owned(self, owner):
        return {n: p for n, p in self.pods.items() if p["owner"] == owner}


def reconcile(cluster, name):
    """一次 reconcile：只看当前状态，不依赖"上次做了什么"（幂等）"""
    spec = cluster.deployments[name]
    pods = cluster.owned(name)
    stale = [n for n, p in pods.items() if p["image"] != spec["image"]]
    if len(pods) < spec["replicas"]:
        cluster.create_pod(name, spec["image"])            # 少了就补
    elif len(pods) > spec["replicas"]:
        cluster.delete_pod(sorted(pods)[-1])               # 多了就删
    elif stale:
        cluster.delete_pod(sorted(stale)[0])               # 数量够了但版本旧：换掉一个
    else:
        return False                                        # 已经收敛，不需要动作
    return True


def run(cluster, name, max_steps=50):
    """控制循环：反复 reconcile 直到不再产生动作"""
    steps = 0
    while steps < max_steps and reconcile(cluster, name):
        for pod in cluster.pods.values():                  # 模拟 kubelet：Pending 的 Pod 过一会变 Running
            if pod["phase"] == "Pending":
                pod["phase"] = "Running"
        steps += 1
    return steps
```

```python
from reconcile import Cluster, reconcile, run

c = Cluster()
c.deployments["llm"] = {"replicas": 3, "image": "v1"}
print("扩到 3 个副本：", run(c, "llm"), "步", sorted(c.owned("llm")))

c.deployments["llm"]["replicas"] = 1
print("缩到 1 个副本：", run(c, "llm"), "步", sorted(c.owned("llm")))

c.deployments["llm"].update(replicas=2, image="v2")
run(c, "llm")
print("换版本到 v2：  ", {n: p["image"] for n, p in sorted(c.owned("llm").items())})

# 有人手动删掉一个 Pod：控制器会自己补回来，这就是"自愈"
victim = sorted(c.owned("llm"))[0]
c.delete_pod(victim)
print(f"手动删掉 {victim} 之后：", run(c, "llm"), "步补回", len(c.owned("llm")), "个")
print("\n事件流水（前 12 条）：", c.events[:12])
print("要点：reconcile 只根据当前状态决定下一步，重复调用不会出错（幂等），所以控制器可以随时重启。")
```

```text title="输出"
扩到 3 个副本： 3 步 ['llm-1', 'llm-2', 'llm-3']
缩到 1 个副本： 2 步 ['llm-1']
换版本到 v2：   {'llm-4': 'v2', 'llm-5': 'v2'}
手动删掉 llm-4 之后： 1 步补回 2 个

事件流水（前 12 条）： ['create llm-1', 'create llm-2', 'create llm-3', 'delete llm-3', 'delete llm-2', 'create llm-4', 'delete llm-1', 'create llm-5', 'delete llm-4', 'create llm-6']
要点：reconcile 只根据当前状态决定下一步，重复调用不会出错（幂等），所以控制器可以随时重启。
```

几个关键结论都在输出里：

- **扩缩容、换版本、自愈，用的是同一个循环**。手动删掉一个 Pod，下一次 reconcile 发现数量少了就补一个——这就是"自愈"，没有任何特殊逻辑。
- **reconcile 必须幂等**。它只看当前状态，所以重复调用、控制器重启、事件重复送达都不会出错。真实的 Kubernetes 还会定期**全量 resync**，专门用来兜住漏掉的事件。
- **一次只做一步**。控制器不会一口气把状态改到位，而是做一个动作、写回状态、等下一次事件。这让整个系统可以被观测、被中断、被限速。

真实控制器还多两层：**informer**（本地缓存 + 监听 watch 事件，避免每次都去 API server 全量读）和 **workqueue**（去重、限速、失败重试）。写 Operator 时这两样由 controller-runtime 提供，你只需要填 `Reconcile()` 函数。

![图：Kubernetes 的控制器模式——声明期望状态，控制器不断比较期望与实际](../assets/figures/k8s-reconcile.svg){.aig-svg}

## 推理服务会用到的对象

| 对象 | 用来干什么 | 推理场景 |
| --- | --- | --- |
| **Pod** | 一组共享网络和卷的容器，调度的最小单位 | 一个推理实例（可能还带日志、指标 sidecar） |
| **Deployment** | 无状态副本集，支持滚动更新与回滚 | 单卡或单机多卡的模型服务 |
| **StatefulSet** | 有稳定名字和存储的副本集，按序启停 | 需要固定身份的场景（少数分布式推理） |
| **Job / CronJob** | 跑完就结束的任务 | 压测、离线批量推理、精度评测 |
| **Service** | 一组 Pod 的稳定入口（ClusterIP / Headless） | 网关到实例的四层负载均衡；Headless 用于多机成员发现 |
| **Ingress / Gateway API** | 七层入口、路由、TLS | 对外的 `/v1/chat/completions` |
| **ConfigMap / Secret** | 配置与凭据 | 模型参数、路由表、HuggingFace token |
| **PVC / PV** | 持久卷 | 模型权重缓存、KV 卸载的磁盘层 |
| **HPA / KEDA** | 按指标扩缩容 | 按队列长度、TTFT、GPU 利用率扩容 |
| **PodDisruptionBudget** | 限制自愿驱逐的并发数 | 升级节点时保证至少 N 个实例在线 |
| **CRD + Operator** | 自定义对象与控制器 | LeaderWorkerSet、Volcano 的 Queue、自家的 InferenceService |

Deployment 与 Pod 之间还夹着 **ReplicaSet**：

```bash
kubectl describe deploy vllm | head -14
```

```text title="输出（本机示例）"
Name:                   vllm
Namespace:              default
CreationTimestamp:      Wed, 30 Sep 2026 11:12:25 +0000
Labels:                 app=vllm
Annotations:            deployment.kubernetes.io/revision: 1
Selector:               app=vllm
Replicas:               3 desired | 3 updated | 3 total | 3 available | 0 unavailable
StrategyType:           RollingUpdate
MinReadySeconds:        0
RollingUpdateStrategy:  25% max unavailable, 25% max surge
```

改一次镜像，Deployment 就新建一个 ReplicaSet，然后按 `maxSurge` / `maxUnavailable` 的节奏此消彼长（下一章展开）。回滚（`kubectl rollout undo`）就是把旧 ReplicaSet 的副本数加回去——旧 ReplicaSet 默认保留 10 个版本，这也是"为什么集群里有一堆 0 副本的 ReplicaSet"的答案。

## Pod 内部：容器、init 与探针

一个推理 Pod 的典型结构：

```yaml
spec:
  initContainers:
    - name: fetch-weights              # 先把权重从对象存储拉到本地盘，拉完才起主容器
      image: rclone/rclone:1.68
      args: ["copy", "s3:models/qwen3-0.6b", "/models/qwen3-0.6b"]
      volumeMounts: [{name: models, mountPath: /models}]
  containers:
    - name: server                     # 主容器：推理引擎
      image: vllm/vllm-openai:v0.11.0
      volumeMounts: [{name: models, mountPath: /models}]
    - name: metrics-proxy              # sidecar：转发/加工指标
      image: nginx:1.27-alpine
  volumes:
    - name: models
      emptyDir: {}                     # 同一个 Pod 内的容器共享这个卷
```

- **共享的**：网络命名空间（同一个 Pod IP，容器之间用 `localhost` 通信）、挂载的卷、生命周期（Pod 被删，里面的容器一起没）。
- **不共享的**：文件系统根目录、进程空间（除非显式开 `shareProcessNamespace`）。
- **initContainer** 按顺序执行、跑完退出，适合"下载权重""检查 GPU 驱动""预热页缓存"这类前置工作。它失败会一直重试，Pod 停在 `Init:0/1`。
- **sidecar** 和主容器并存。1.29 起有原生的 sidecar（写成 `restartPolicy: Always` 的 initContainer），它先于主容器启动、后于主容器停止，比传统 sidecar 更适合日志和代理。

## requests、limits 与 QoS

```yaml
resources:
  requests: {cpu: "8", memory: 32Gi, nvidia.com/gpu: 1}
  limits:   {cpu: "16", memory: 64Gi, nvidia.com/gpu: 1}
```

- **requests 决定调度**：调度器按节点"可分配量减去已被 requests 占用的量"来判断放不放得下，和实际用了多少无关。
- **limits 决定运行时**：CPU 超了被 cgroup 节流（不会被杀，但延迟飙升，见[计算机基础：容器](root://cs/os/containers/)）；内存超了直接 OOMKill。
- **GPU 只能整数、且 requests 必须等于 limits**：`nvidia.com/gpu` 是不可压缩的扩展资源，不支持小数（要共享一张卡得用 MIG 或时间片，见第三章）。
- **QoS 三档**：requests == limits 且都写了是 `Guaranteed`（最不容易被驱逐），只写 requests 是 `Burstable`，都不写是 `BestEffort`（节点内存紧张时最先被杀）。推理服务应该是 Guaranteed。

一个常见的坑：**把 CPU limits 设得很小**。推理进程的 CPU 开销不只是"调度器那点逻辑"——分词、采样、HTTP、指标、以及 PyTorch 的线程池都要 CPU，被节流的表现是 TTFT 抖动、GPU 利用率上不去，而监控上看 CPU"没用满"（因为被节流了）。

## 排障的第一组命令

```bash
kubectl get pods -l app=vllm -o wide          # 状态、重启次数、在哪个节点、Pod IP
kubectl describe pod <name>                   # Events 在最下面，90% 的原因都在这里
kubectl logs <name> --previous                # 上一个容器实例的日志（崩溃重启时最有用）
kubectl get events --sort-by=.lastTimestamp   # 全命名空间的事件流
kubectl exec -it <name> -- nvidia-smi         # 进容器看 GPU
kubectl top pod / kubectl top node            # 实时资源用量（需要 metrics-server）
```

按 Pod 状态定位：

| 状态 | 常见原因 |
| --- | --- |
| `Pending` | 没有节点满足 requests（GPU 不够、污点/容忍不匹配、亲和性冲突），看 `describe` 里的 `FailedScheduling` |
| `ContainerCreating` | 拉镜像慢、挂载卷失败、device plugin 没就绪 |
| `CrashLoopBackOff` | 容器起来就退出，看 `logs --previous`：参数错、显存不足、权重路径不对 |
| `OOMKilled`（在 `describe` 的 Last State 里） | 内存 limits 太小；注意权重加载时的峰值和 `/dev/shm` |
| `Running` 但 `0/1 READY` | 就绪探针没通过——权重还在加载，或者探针路径/超时设得不对 |
| `Terminating` 很久 | 优雅退出时间长（正在处理的请求没做完），或者进程不响应 SIGTERM |

!!! interview "面试怎么答"
    被问 Kubernetes：先讲内核——声明式 API 加控制器循环。apply 只是写对象，真正干活的是各控制器的 reconcile：读期望、读实际、做一步动作；必须幂等，因为控制器会重启、事件会重复或丢失，所以还有定期 resync。由此解释自愈和滚动更新是同一套机制：Deployment 调整新旧两个 ReplicaSet 的副本数，节奏由 maxSurge/maxUnavailable 控制。再讲推理服务落到哪些对象上（Deployment、Job、Service、PVC、HPA、PDB、CRD），以及 requests/limits 的区别：requests 决定调度、limits 决定节流与 OOMKill，GPU 必须整数且 requests == limits，服务要做成 Guaranteed。最后给排障套路：先 `get -o wide` 看状态，再 `describe` 看事件，`logs --previous` 看崩溃原因，按 Pending / CrashLoopBackOff / 0/1 READY 分别对应调度、启动、探针三类问题。

## 练习

**1. apply 成功但没有 Pod。** 执行 `kubectl apply -f deploy.yaml` 返回 `deployment.apps/vllm created`，但 `kubectl get pods` 是空的。列出三种可能的原因和对应的排查命令。

??? success "参考答案"
    （1）**ReplicaSet 没造出来**：`kubectl get rs -l app=vllm`、`kubectl describe deploy vllm` 看事件，常见原因是 `selector` 和 `template.metadata.labels` 不匹配（API server 会直接拒绝）、或者被准入控制器（如 OPA/Kyverno 策略）拦了；（2）**Pod 造出来了但被另一个命名空间隔开**：确认 `-n` 参数和当前 context；（3）**配额不足**：`kubectl describe quota`、`kubectl get events`，ResourceQuota 超了会让 ReplicaSet 一直创建失败，事件里是 `exceeded quota`。

**2. 为什么控制器要幂等。** 假设 reconcile 写成"记住上次创建了几个，这次只补差额"，控制器重启后会发生什么？

??? success "参考答案"
    重启后内存里的"上次创建了几个"丢了，如果默认从 0 开始，它会把副本数再创建一遍，造成双倍的 Pod；如果状态存在别处又可能和实际不一致（比如有人手动删了 Pod）。正确的做法是每次都**重新观测实际状态**（列出属于我的 Pod 有几个），再决定动作——这样无论重启多少次、事件重复多少次，结果都收敛到期望状态。

**3. 设计资源规格。** 一个推理实例要一张 H100、峰值内存 40 GB（加载权重时会到 55 GB）、CPU 平时 4 核、组 batch 和分词时会冲到 12 核。写出 `resources`，并说明为什么。

??? success "参考答案"
    ```yaml
    resources:
      requests: {cpu: "8", memory: 64Gi, nvidia.com/gpu: 1}
      limits:   {cpu: "16", memory: 64Gi, nvidia.com/gpu: 1}
    ```
    内存 requests = limits = 64 Gi：按**加载峰值**而不是稳态设置，并留一点余量，否则加载权重时就被 OOMKill；两者相等让它进 Guaranteed 档，不容易被驱逐。CPU requests 取一个稳态偏上的值（8）保证调度到足够空闲的节点，limits 给到 16 允许突发——CPU 是可压缩资源，超了只是被节流。GPU 必须整数且 requests 与 limits 一致。另外别忘了 `/dev/shm`：很多推理框架用共享内存做进程间通信，默认只有 64 MB，要用 `emptyDir: {medium: Memory}` 挂大一点。

**4. 读懂 ReplicaSet。** `kubectl get rs` 显示两个 ReplicaSet：`vllm-6ccd` 期望 3 个、`vllm-7f9a` 期望 0 个。这说明什么？如果想回到 `vllm-7f9a` 的版本，命令是什么？

??? success "参考答案"
    说明做过一次滚动更新：`vllm-6ccd` 是当前版本，`vllm-7f9a` 是上一个版本（副本数被降到 0，但对象保留着，用于回滚和查历史）。回滚用 `kubectl rollout undo deploy/vllm`（回到上一个版本）或 `kubectl rollout undo deploy/vllm --to-revision=N`；`kubectl rollout history deploy/vllm` 看版本列表。保留几个历史版本由 `spec.revisionHistoryLimit` 控制（默认 10）。

## 小结

- [x] 声明式 API：apply 只是写对象，控制器的 reconcile 循环把现实拉向期望；所以 apply 成功 ≠ 服务可用。
- [x] reconcile 只看当前状态、一次做一步、必须幂等；informer 管缓存、workqueue 管去重与重试。
- [x] Deployment → ReplicaSet → Pod；滚动更新和回滚都是在调两个 ReplicaSet 的副本数。
- [x] initContainer 做前置工作（拉权重），sidecar 做旁路（日志、指标、代理），同 Pod 共享网络和卷。
- [x] requests 决定调度、limits 决定节流与 OOMKill；GPU 必须整数且两者相等；推理服务要做成 Guaranteed，别忘了 `/dev/shm`。
- [x] 排障顺序：`get -o wide` → `describe`（看 Events）→ `logs --previous`；Pending 看调度、CrashLoop 看日志、0/1 READY 看探针。
