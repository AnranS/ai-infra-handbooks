# 部署形态：探针、滚动发布与扩缩容

<p class="lead">推理服务和普通 Web 服务在 Kubernetes 上的差别，几乎全部来自两个数字：**启动要几分钟**（加载几十 GB 权重）和**退出要几分钟**（把手上的长回答做完）。这两个数字会放大所有默认配置的问题——探针把还在加载权重的实例判成"挂了"、滚动发布卡上十几分钟、HPA 扩容永远慢半拍、节点维护一次带走一半容量。这一章用两个模拟把这些账算清楚，并给出推理服务该怎么配探针、发布策略、HPA/KEDA 和 PDB。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 存活、就绪、启动三种探针分别管什么？推理服务漏配哪一个会最惨？
    2. `maxSurge` 和 `maxUnavailable` 各是什么意思？对 GPU 服务意味着什么代价？
    3. Pod 收到 SIGTERM 之后会发生什么？为什么推理服务要调大 `terminationGracePeriodSeconds`？
    4. HPA 的扩容公式是什么？为什么按 GPU 利用率扩容常常不好使？
    5. PodDisruptionBudget 拦得住什么、拦不住什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. **存活探针**（liveness）失败会**重启容器**；**就绪探针**（readiness）失败只是把 Pod 从 Service 的端点里摘掉，不重启；**启动探针**（startup）在启动阶段接管前两者，给慢启动留出时间。推理服务漏配 startup 最惨：权重还在加载，liveness 就判定失败，容器被反复重启，永远起不来（表现为 CrashLoopBackOff）。
    2. `maxSurge` 是发布期间允许超出期望副本数多少（要额外的 GPU），`maxUnavailable` 是允许少多少（牺牲容量）。GPU 服务上这是真金白银：`maxSurge=1` 就要多占一份卡，`maxUnavailable=1` 就要少一份容量。本章的模拟里，6 副本、新副本 60 秒就绪时，`(1,0)` 要 540 秒、`(2,1)` 要 190 秒、`(6,0)`（蓝绿）只要 115 秒但要双倍的卡。
    3. kubelet 先把 Pod 从 Service 端点里摘掉（新请求不再进来），同时给容器发 SIGTERM，然后等 `terminationGracePeriodSeconds`，超时才 SIGKILL。推理服务手上可能有正在生成的长回答，这个时间要按"最长回答的生成时间"设置（几十秒到几分钟），否则用户会看到回答被截断。
    4. `期望副本数 = 当前副本数 × (当前指标 / 目标指标)`，偏差在容忍带（默认 10%）内不动。GPU 利用率不好使的原因：它是"这一时刻有没有 kernel 在跑"，decode 阶段哪怕只有一个请求也能让利用率接近 100%，区分不出忙闲。更好的指标是**待处理请求数 / 队列长度 / TTFT / 每副本并发**，用 KEDA 或自定义指标适配器接进来。
    5. PDB 限制的是**自愿驱逐**（`kubectl drain`、节点升级、集群自动缩容），保证同时被驱逐的副本数不超过预算。它拦不住**非自愿中断**：节点宕机、OOMKill、以及高优先级 Pod 的抢占。

## 三种探针：别让权重加载被判成故障

```yaml
containers:
  - name: server
    image: vllm/vllm-openai:v0.11.0
    startupProbe:                     # 启动阶段专用：最多容忍 30 × 10 = 300 秒
      httpGet: {path: /health, port: 8000}
      periodSeconds: 10
      failureThreshold: 30
    readinessProbe:                   # 就绪：决定要不要往这个实例发流量
      httpGet: {path: /health, port: 8000}
      periodSeconds: 5
      failureThreshold: 3
    livenessProbe:                    # 存活：只抓"进程卡死"，判定要宽松
      httpGet: {path: /health, port: 8000}
      periodSeconds: 20
      failureThreshold: 6
```

三条经验：

- **startupProbe 是推理服务的必需品**。权重加载几分钟很常见（从对象存储拉、反量化、编译 CUDA Graph）。有了它，前两种探针在启动期间不生效，`failureThreshold × periodSeconds` 就是允许的最长启动时间。
- **liveness 要比 readiness 宽松得多**。一个正在处理大 batch 的实例可能几秒内不响应探针，这不是故障；如果 liveness 太敏感，它会在高负载时把健康实例一个个重启，把负载推给剩下的实例——**雪崩**。经验是 liveness 只用来抓真正的死锁，周期长、阈值大。
- **readiness 可以用来做背压**。实例队列太长、KV Cache 快满时主动让 `/health` 返回失败，负载均衡器就会暂时不给它发新请求（见[计算机基础：负载均衡与排队](root://cs/net/load-balance/)）。注意要加滞后，否则所有实例会一起抖动。

## 滚动发布：三个参数决定代价

![图：滚动发布——新旧 ReplicaSet 此消彼长，探针决定就绪与存活](../assets/figures/rolling-update.svg){.aig-svg}

```bash
kubectl set image deploy/vllm server=vllm/vllm-openai:v0.11.1
kubectl rollout status deploy/vllm
kubectl get rs -l app=vllm --sort-by=.metadata.creationTimestamp
```

```text title="输出（本机示例）"
Waiting for deployment "vllm" rollout to finish: 1 out of 3 new replicas have been updated...
Waiting for deployment "vllm" rollout to finish: 2 out of 3 new replicas have been updated...

NAME              DESIRED   CURRENT   READY   AGE
vllm-6ccd564c7d   1         1         1       10m
vllm-67dfd66c87   3         3         2       14s
```

可以清楚看到"新旧两个 ReplicaSet 此消彼长"：旧的从 3 降到 1，新的从 0 升到 3。节奏由两个参数控制，代价可以算出来：

```python title="rollout.py"
"""滚动更新与自动扩缩容的时间线模拟。

滚动更新：Deployment 按 maxSurge / maxUnavailable 调整新旧两个 ReplicaSet 的副本数，
推理服务的特殊之处是"就绪"很慢（要加载几十 GB 权重）、"退出"也很慢（要把手上的请求做完）。
"""


def rollout(replicas, ready_s, drain_s, max_surge, max_unavailable, step_s=5, horizon_s=600):
    """返回 [(时刻, 可用副本数, 总副本数), ...]，以及整个发布耗时"""
    old = [{"ready": True} for _ in range(replicas)]      # 旧版本：都已就绪
    new = []                                              # 新版本：正在启动
    timeline, t = [], 0
    while t <= horizon_s:
        for pod in new:                                   # 新 Pod 到点就绪
            if not pod["ready"] and t >= pod["at"] + ready_s:
                pod["ready"] = True
        old = [p for p in old if not (p.get("draining") and t >= p["drain_at"] + drain_s)]
        available = sum(1 for p in old if p["ready"] and not p.get("draining")) + \
                    sum(1 for p in new if p["ready"])
        total = len(old) + len(new)
        timeline.append((t, available, total))
        if not old and all(p["ready"] for p in new) and len(new) == replicas:
            break
        if total < replicas + max_surge and len(new) < replicas:          # 还能再起新的
            new.append({"ready": False, "at": t})
        elif available - 1 >= replicas - max_unavailable:                  # 可以开始撤一个旧的
            for pod in old:
                if not pod.get("draining"):
                    pod["draining"], pod["drain_at"] = True, t
                    break
        t += step_s
    return timeline, t


def summarize(timeline, replicas):
    """发布过程中最少可用了几个副本、低于目标容量的时间有多久"""
    worst = min(a for _, a, _ in timeline)
    degraded = sum(1 for _, a, _ in timeline if a < replicas)
    return worst, degraded


def hpa_step(current, desired_metric, target_metric, replicas, min_r, max_r, tolerance=0.1):
    """HPA 的核心公式：期望副本数 = 当前副本数 × (当前指标 / 目标指标)，偏差在容忍带内就不动"""
    ratio = desired_metric / target_metric
    if abs(ratio - 1) <= tolerance:
        return replicas
    return max(min_r, min(max_r, -(-int(replicas * ratio * 100) // 100)))


def simulate_hpa(load, replicas, target_qps_per_pod, ready_s, step_s=15, min_r=1, max_r=20):
    """load 是每一步的总 QPS；返回 [(步, 负载, 副本数, 每副本 QPS, 是否过载)]"""
    pending = []                                       # 正在启动的副本：(就绪时刻, 数量)
    out = []
    for i, qps in enumerate(load):
        t = i * step_s
        ready_now = sum(n for at, n in pending if t >= at)
        pending = [(at, n) for at, n in pending if t < at]
        replicas += ready_now
        per_pod = qps / replicas if replicas else float("inf")
        out.append((t, qps, replicas, round(per_pod, 1), per_pod > target_qps_per_pod * 1.2))
        want = hpa_step(qps, per_pod, target_qps_per_pod, replicas, min_r, max_r)
        if want > replicas:
            pending.append((t + ready_s, want - replicas))   # 扩容要等启动
        elif want < replicas:
            replicas = want                                   # 缩容立刻生效（真实 HPA 还有冷却窗口）
    return out
```

```python
from rollout import rollout, simulate_hpa, summarize

print("滚动更新：6 个副本，新副本 60 秒才就绪（加载权重），旧副本 30 秒优雅退出")
for surge, unavail in ((1, 0), (2, 1), (6, 0)):
    timeline, total = rollout(6, ready_s=60, drain_s=30, max_surge=surge, max_unavailable=unavail)
    worst, degraded = summarize(timeline, 6)
    extra = "需要额外 %d 份 GPU" % surge
    print(f"  maxSurge={surge}, maxUnavailable={unavail}：发布耗时 {total:3d} 秒，"
          f"过程中最少可用 {worst} 个，{extra}")

print("\n就绪探针配错的代价（maxSurge=2, maxUnavailable=1）：")
for ready_s in (10, 60, 180):
    timeline, total = rollout(6, ready_s=ready_s, drain_s=30, max_surge=2, max_unavailable=1)
    print(f"  新副本 {ready_s:3d} 秒就绪：整个发布 {total:3d} 秒")
```

```text title="输出"
滚动更新：6 个副本，新副本 60 秒才就绪（加载权重），旧副本 30 秒优雅退出
  maxSurge=1, maxUnavailable=0：发布耗时 540 秒，过程中最少可用 6 个，需要额外 1 份 GPU
  maxSurge=2, maxUnavailable=1：发布耗时 190 秒，过程中最少可用 5 个，需要额外 2 份 GPU
  maxSurge=6, maxUnavailable=0：发布耗时 115 秒，过程中最少可用 6 个，需要额外 6 份 GPU

就绪探针配错的代价（maxSurge=2, maxUnavailable=1）：
  新副本  10 秒就绪：整个发布  95 秒
  新副本  60 秒就绪：整个发布 190 秒
  新副本 180 秒就绪：整个发布 430 秒
```

第一段是发布策略的三种典型取舍：

| 策略 | 发布耗时 | 额外 GPU | 容量损失 |
| --- | --- | --- | --- |
| `maxSurge=1, maxUnavailable=0` | 最慢（一个一个换） | 1 份 | 无 |
| `maxSurge=2, maxUnavailable=1` | 中等 | 2 份 | 1 份 |
| `maxSurge=replicas, maxUnavailable=0`（蓝绿） | 最快 | 双倍 | 无 |

GPU 服务通常没有"双倍的卡"，所以实践中要么接受慢（在低峰期发布），要么容忍一点容量损失。第二段说明**启动时间直接乘进发布耗时**：新副本 10 秒就绪时整个发布 95 秒，180 秒就绪时要 430 秒——这就是为什么要认真优化权重加载（本地缓存、预拉镜像、mmap 加载，见[生产部署与运维](../ops/deploy.md)）。

更稳的发布方式：

- **金丝雀**：先发一个新版本实例，用 Service 的标签选择器或网关按比例切一小部分流量，观察指标（TTFT、错误率、token 吞吐）再继续。用 Argo Rollouts / Flagger 可以自动化这个过程。
- **影子流量**：把线上请求复制一份给新版本，不返回给用户，专门用来对比输出和性能。模型换版本时特别有用——可以直接比对两个版本的生成结果。
- **`minReadySeconds`**：新 Pod 就绪后再观察几秒才算数，避免"刚就绪就崩"的实例被算进可用副本。

## 优雅退出：别把用户的回答截断

```yaml
spec:
  terminationGracePeriodSeconds: 180     # 按最长回答的生成时间设置
  containers:
    - name: server
      lifecycle:
        preStop:
          exec:
            command: ["sh", "-c", "sleep 5"]   # 给端点摘除留出传播时间
```

停止一个 Pod 的完整时序：

1. Pod 被标记为 Terminating，**同时**发生两件事：从所有 Service 的端点列表里摘掉（新请求不再来），以及执行 `preStop` 钩子；
2. `preStop` 结束后，容器收到 **SIGTERM**；
3. 等待 `terminationGracePeriodSeconds`；
4. 还没退出就 **SIGKILL**。

两个坑：

- **端点摘除是异步的**。kube-proxy、Ingress、网关各自更新规则需要时间，这期间新请求还会进来。`preStop` 里 sleep 几秒（或让 readiness 先失败）能盖住这个窗口。
- **进程要真的处理 SIGTERM**。推理框架要停止接新请求、把队列里的请求做完、释放显存再退出。如果进程忽略 SIGTERM，那就只能等 SIGKILL，正在生成的回答直接断掉。vLLM、SGLang 都实现了优雅退出，自己包一层服务时要记得转发信号（容器里 PID 1 的坑：用 `exec` 启动或加 `tini`）。

## 扩缩容：指标选错就白扩

HPA 的公式很简单，难的是选指标：

```python
from rollout import simulate_hpa

print("HPA：目标每副本 10 QPS，副本 90 秒才能就绪，流量在第 5 步翻三倍")
load = [30] * 5 + [90] * 12 + [30] * 6
for ready_s in (90, 15):
    rows = simulate_hpa(load, replicas=3, target_qps_per_pod=10, ready_s=ready_s)
    overloaded = sum(1 for *_, bad in rows if bad)
    print(f"  副本 {ready_s:2d} 秒就绪：过载的采样点 {overloaded:2d} 个 / {len(rows)}")
    for t, qps, replicas, per_pod, bad in rows[3:11]:
        print(f"    t={t:3d}s 负载 {qps:3d} QPS，副本 {replicas:2d}，每副本 {per_pod:5.1f} QPS"
              f"{'  ← 过载' if bad else ''}")
```

```text title="输出"
HPA：目标每副本 10 QPS，副本 90 秒才能就绪，流量在第 5 步翻三倍
  副本 90 秒就绪：过载的采样点  6 个 / 23
    t= 45s 负载  30 QPS，副本  3，每副本  10.0 QPS
    t= 60s 负载  30 QPS，副本  3，每副本  10.0 QPS
    t= 75s 负载  90 QPS，副本  3，每副本  30.0 QPS  ← 过载
    t= 90s 负载  90 QPS，副本  3，每副本  30.0 QPS  ← 过载
    t=105s 负载  90 QPS，副本  3，每副本  30.0 QPS  ← 过载
    t=120s 负载  90 QPS，副本  3，每副本  30.0 QPS  ← 过载
    t=135s 负载  90 QPS，副本  3，每副本  30.0 QPS  ← 过载
    t=150s 负载  90 QPS，副本  3，每副本  30.0 QPS  ← 过载
  副本 15 秒就绪：过载的采样点  1 个 / 23
    t= 45s 负载  30 QPS，副本  3，每副本  10.0 QPS
    t= 60s 负载  30 QPS，副本  3，每副本  10.0 QPS
    t= 75s 负载  90 QPS，副本  3，每副本  30.0 QPS  ← 过载
    t= 90s 负载  90 QPS，副本  9，每副本  10.0 QPS
    t=105s 负载  90 QPS，副本  9，每副本  10.0 QPS
    t=120s 负载  90 QPS，副本  9，每副本  10.0 QPS
    t=135s 负载  90 QPS，副本  9，每副本  10.0 QPS
    t=150s 负载  90 QPS，副本  9，每副本  10.0 QPS
```

模拟说明了两件事：**扩容永远慢半拍**（要等新副本就绪，推理服务尤其慢），以及**启动时间决定了过载持续多久**（90 秒就绪时有 6 个采样点过载，15 秒就绪时只有 1 个）。

所以推理服务的扩缩容要这样配：

| 做法 | 原因 |
| --- | --- |
| 指标用**待处理请求数 / 队列长度 / 每副本并发**，不用 GPU 利用率 | decode 阶段一个请求也能把利用率跑满，区分不出忙闲 |
| 用 **KEDA** 或自定义指标适配器接 Prometheus | HPA 原生只认 CPU/内存和 custom metrics API |
| `scaleUp` 激进、`scaleDown` 保守（`behavior` 里配） | 扩容慢半拍要提前，缩容太快会在流量抖动时反复启停 |
| 保留**预热的副本池**或用低优先级占位 Pod | 真正要扩容时抢占占位 Pod，省掉调度和拉镜像的时间 |
| 结合 **Cluster Autoscaler / Karpenter** | Pod 扩不上去是因为节点不够时，要连节点一起扩，这又多几分钟 |

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: {name: vllm}
spec:
  scaleTargetRef: {name: vllm}
  minReplicaCount: 2
  maxReplicaCount: 20
  cooldownPeriod: 300                 # 缩容前的冷却，避免抖动
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        query: sum(vllm:num_requests_waiting)    # 排队中的请求数
        threshold: "5"                            # 每个副本平均排队 5 个就扩
```

## PodDisruptionBudget：别让运维一次拿走半个集群

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: {name: vllm}
spec:
  minAvailable: 80%                   # 或 maxUnavailable: 1
  selector: {matchLabels: {app: vllm}}
```

`kubectl drain` 一个节点、集群自动缩容、节点升级，这些**自愿驱逐**都会先问 PDB："再赶走一个，还满足预算吗？"不满足就等着。它拦不住节点宕机、OOMKill 和抢占。

对推理服务，PDB 的值要和发布策略一起想：如果 `maxUnavailable=1` 而 PDB 写 `minAvailable: 100%`，节点维护会直接卡死。常见做法是 `maxUnavailable: 1` 加上足够的副本数，让运维总能一台一台地滚。

!!! interview "面试怎么答"
    被问推理服务怎么部署：先点出两个特殊数字——启动几分钟、退出几分钟，再逐项讲。探针：startupProbe 必配（否则权重加载被 liveness 判成故障，反复重启），liveness 要比 readiness 宽松得多（否则高负载时重启健康实例引发雪崩），readiness 可以拿来做背压。发布：maxSurge 要额外的卡、maxUnavailable 牺牲容量，GPU 服务通常只能慢滚或低峰发布，金丝雀和影子流量更稳；启动时间直接乘进发布耗时。退出：端点摘除是异步的，要用 preStop 盖住窗口，terminationGracePeriodSeconds 按最长回答设置，进程要真的处理 SIGTERM。扩缩容：别用 GPU 利用率（decode 一个请求就能跑满），用队列长度或每副本并发，KEDA 接 Prometheus，扩容激进缩容保守，配合节点扩容和预热池。最后补 PDB：只管自愿驱逐，要和发布策略一起设计。

## 练习

**1. 算发布窗口。** 一个 12 副本的服务，新副本就绪要 150 秒，优雅退出 60 秒。`maxSurge=2, maxUnavailable=1` 时整个发布大约多久？如果集群只能多给 1 份 GPU 呢？

??? success "参考答案"
    用本章的 `rollout()` 跑一遍即可（把 `replicas=12, ready_s=150, drain_s=60` 代入）。直觉估算：每"一批"能同时替换 3 个（surge 2 + unavailable 1），12 个副本要 4 批，每批的关键路径是"新副本就绪 150 秒"，所以大约 4 × 150 = 600 秒起步，加上退出的重叠部分。

    只能多给 1 份 GPU 时（`maxSurge=1`），并行度降到 2，批数翻倍，发布时间接近翻倍。这时更好的选择是：低峰期发布、或者先临时扩容一批节点再发布、或者接受 `maxUnavailable=2` 牺牲一点容量。

**2. 诊断反复重启。** 一个推理 Pod 一直 CrashLoopBackOff，`kubectl logs --previous` 显示日志停在"Loading model weights..."，没有报错。最可能的原因是什么？

??? success "参考答案"
    liveness 探针在权重加载完成前就判定失败，kubelet 重启了容器；重启后又从头加载，循环往复。确认方法：`kubectl describe pod` 里会有 `Liveness probe failed` 的事件，容器的 `Last State` 是 `Terminated`、`Reason: Error` 而不是应用自己的报错。

    修法：加 `startupProbe`（`failureThreshold × periodSeconds` 要大于最长加载时间），并把 liveness 的 `initialDelaySeconds` / `failureThreshold` 放宽。顺带检查加载慢的原因（镜像里没有权重、每次都从对象存储拉、没有本地缓存卷）。

**3. 选扩容指标。** 一个服务的 GPU 利用率长期在 95%，但 p99 TTFT 从 200 ms 涨到 3 秒。用 GPU 利用率做 HPA 指标会发生什么？该用什么？

??? success "参考答案"
    什么也不会发生——利用率本来就 95%，HPA 认为"已经达到目标"，不扩容。但 TTFT 涨了 15 倍，说明请求在排队。

    应该用**排队指标**：`num_requests_waiting`（vLLM/SGLang 都有这个指标）、每副本并发数、或者直接用 TTFT 的分位数。用 KEDA 接 Prometheus 查询，阈值按"每副本平均排队几个"设置。同时把 `scaleUp` 的 `stabilizationWindowSeconds` 调小、`scaleDown` 调大。

**4. 设计一次安全的模型换版本。** 要把线上的 v1 模型换成 v2（权重不同、输出会变），服务有 20 个副本、SLO 是 p99 TTFT < 1 秒。写出发布步骤。

??? success "参考答案"
    （1）**影子流量**：先起 2 个 v2 副本，复制一份线上请求给它们（不返回给用户），对比输出差异和性能指标，跑够样本；（2）**金丝雀**：把 5% 流量切给 v2（用网关按比例路由或多一个 Deployment + 权重），观察 TTFT、错误率、生成质量的线上指标 30 分钟；（3）**逐步放量**：5% → 25% → 50% → 100%，每一档都留观察窗口，指标异常就立刻切回；（4）全量后保留 v1 的 ReplicaSet 一段时间，便于 `kubectl rollout undo` 快速回滚；（5）整个过程配 PDB 和低峰期窗口，并提前确认有足够的卡承载两个版本并存。

## 小结

- [x] 推理服务的两个特殊数字：启动几分钟、退出几分钟，所有默认配置都要按它们重算。
- [x] startupProbe 必配；liveness 要宽松（否则高负载时雪崩）；readiness 可以做背压。
- [x] 滚动发布的代价是"额外的卡（maxSurge）或损失的容量（maxUnavailable）"，启动时间直接乘进发布耗时；更稳的是金丝雀和影子流量。
- [x] 优雅退出：端点摘除是异步的（用 preStop 盖窗口），宽限期按最长回答设置，进程要真的处理 SIGTERM。
- [x] 扩容指标用排队长度而不是 GPU 利用率；扩容慢半拍，要激进扩、保守缩，配合节点扩容与预热池。
- [x] PDB 只管自愿驱逐，要和发布策略一起设计。
