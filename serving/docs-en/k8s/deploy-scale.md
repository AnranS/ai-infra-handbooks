# Deployment patterns: probes, rolling releases and scaling

<p class="lead">Almost all the difference between inference services and ordinary web services on Kubernetes comes from two numbers: **starting takes minutes** (loading tens of GB of weights) and **stopping takes minutes** (finishing the long answers in hand). These two numbers magnify the problems of every default setting: probes judge instances still loading weights as "dead", rolling releases drag on for over ten minutes, HPA scale-ups always lag a step behind, and one round of node maintenance takes away half the capacity. This chapter does the accounting with two simulations, and shows how inference services should configure probes, release strategies, HPA/KEDA and PDBs.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What do the liveness, readiness and startup probes each govern? Which one hurts an inference service most if left out?
    2. What do `maxSurge` and `maxUnavailable` each mean? What do they cost for a GPU service?
    3. What happens after a Pod receives SIGTERM? Why should inference services raise `terminationGracePeriodSeconds`?
    4. What is the HPA's scale-up formula? Why does scaling on GPU utilization often fail?
    5. What can a PodDisruptionBudget stop, and what can't it?

??? success "Answers (try first, then expand to compare)"
    1. A failing **liveness probe** **restarts the container**; a failing **readiness probe** only removes the Pod from the Service's endpoints, without restarting; the **startup probe** takes over from the other two during startup, leaving time for slow starts. Leaving out startup hurts inference services most: while the weights are still loading, liveness judges failure, the container is restarted again and again, and it never comes up (showing as CrashLoopBackOff).
    2. `maxSurge` is how many replicas above the desired count are allowed during a release (extra GPUs needed), and `maxUnavailable` how many below (capacity sacrificed). On GPU services this is real money: `maxSurge=1` takes one more GPU's worth, and `maxUnavailable=1` one less of capacity. In this chapter's simulation, with 6 replicas and new replicas ready in 60 seconds, `(1,0)` takes 540 seconds, `(2,1)` 190 seconds, and `(6,0)` (blue-green) only 115 seconds, but needs double the GPUs.
    3. The kubelet first removes the Pod from the Service endpoints (no new requests come in) while sending SIGTERM to the containers, then waits `terminationGracePeriodSeconds`, and only on timeout sends SIGKILL. An inference service may have long answers mid-generation, so this time should be set by "the generation time of the longest answer" (tens of seconds to minutes), or users see answers cut off.
    4. `desired replicas = current replicas × (current metric / target metric)`, with no change when the deviation is within the tolerance band (10% by default). GPU utilization fails because it measures "whether any kernel is running at this moment": in decode, even a single request can push utilization close to 100%, so it can't tell busy from idle. Better metrics are **pending requests / queue length / TTFT / concurrency per replica**, brought in with KEDA or a custom metrics adapter.
    5. A PDB limits **voluntary evictions** (`kubectl drain`, node upgrades, cluster autoscaler scale-down), ensuring the number of replicas evicted at once stays within budget. It cannot stop **involuntary disruptions**: node crashes, OOMKills, and preemption by higher-priority Pods.

## Three probes: don't let weight loading be judged a failure {#三种探针别让权重加载被判成故障}

```yaml
containers:
  - name: server
    image: vllm/vllm-openai:v0.11.0
    startupProbe:                     # for the startup phase only: tolerates at most 30 × 10 = 300 seconds
      httpGet: {path: /health, port: 8000}
      periodSeconds: 10
      failureThreshold: 30
    readinessProbe:                   # readiness: decides whether to send traffic to this instance
      httpGet: {path: /health, port: 8000}
      periodSeconds: 5
      failureThreshold: 3
    livenessProbe:                    # liveness: only catches "the process is stuck", so judge leniently
      httpGet: {path: /health, port: 8000}
      periodSeconds: 20
      failureThreshold: 6
```

Three lessons:

- **startupProbe is a must for inference services**. Loading weights for several minutes is common (pulling from object storage, dequantizing, compiling CUDA Graphs). With it, the other two probes don't apply during startup, and `failureThreshold × periodSeconds` is the longest startup allowed.
- **liveness must be far more lenient than readiness**. An instance processing a big batch may not answer probes for a few seconds, which is not a failure; if liveness is too sensitive, it restarts healthy instances one by one under high load, pushing the load onto those that remain: an **avalanche**. The rule of thumb is that liveness only catches genuine deadlocks, with a long period and a high threshold.
- **readiness can be used for backpressure**. When an instance's queue is too long or the KV Cache is nearly full, have `/health` fail on purpose, and the load balancer temporarily stops sending it new requests (see [CS Fundamentals: load balancing and queueing](root://cs/net/load-balance/)). Add hysteresis, or every instance flaps together.

## Rolling releases: three parameters decide the cost {#滚动发布三个参数决定代价}

![Figure: a rolling release: the old and new ReplicaSets trade places, and probes decide readiness and liveness](../assets/figures/rolling-update.svg){.aig-svg}

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

You can clearly see "the old and new ReplicaSets trading places": the old one drops from 3 to 1, the new one rises from 0 to 3. The pace is controlled by two parameters, and the cost can be computed:

```python title="rollout.py"
"""滚动更新与自动扩缩容的时间线模拟。

滚动更新：Deployment 按 maxSurge / maxUnavailable 调整新旧两个 ReplicaSet 的副本数，
推理服务的特殊之处是"就绪"很慢（要加载几十 GB 权重）、"退出"也很慢（要把手上的请求做完）。
"""


def rollout(replicas, ready_s, drain_s, max_surge, max_unavailable, step_s=5, horizon_s=600):
    """返回 [(时刻, 可用副本数, 总副本数), ...]，以及整个发布耗时"""
    old = [{"ready": True} for _ in range(replicas)]      # old version: all ready
    new = []                                              # new version: starting
    timeline, t = [], 0
    while t <= horizon_s:
        for pod in new:                                   # new Pods become ready on schedule
            if not pod["ready"] and t >= pod["at"] + ready_s:
                pod["ready"] = True
        old = [p for p in old if not (p.get("draining") and t >= p["drain_at"] + drain_s)]
        available = sum(1 for p in old if p["ready"] and not p.get("draining")) + \
                    sum(1 for p in new if p["ready"])
        total = len(old) + len(new)
        timeline.append((t, available, total))
        if not old and all(p["ready"] for p in new) and len(new) == replicas:
            break
        if total < replicas + max_surge and len(new) < replicas:          # room to start another new one
            new.append({"ready": False, "at": t})
        elif available - 1 >= replicas - max_unavailable:                  # an old one can start draining
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
    pending = []                                       # replicas starting: (ready time, count)
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
            pending.append((t + ready_s, want - replicas))   # scaling up waits for startup
        elif want < replicas:
            replicas = want                                   # scaling down takes effect immediately (the real HPA also has a cooldown window)
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

The first section shows three typical trade-offs in release strategy:

| Strategy | Release time | Extra GPUs | Capacity lost |
| --- | --- | --- | --- |
| `maxSurge=1, maxUnavailable=0` | slowest (replacing one at a time) | 1 | none |
| `maxSurge=2, maxUnavailable=1` | medium | 2 | 1 |
| `maxSurge=replicas, maxUnavailable=0` (blue-green) | fastest | double | none |

GPU services usually don't have "double the GPUs", so in practice you either accept slowness (release at off-peak hours) or tolerate a little capacity loss. The second section shows that **startup time multiplies directly into release time**: with new replicas ready in 10 seconds the whole release takes 95 seconds, and with 180 seconds it takes 430; this is why weight loading deserves serious optimization (local caches, pre-pulled images, mmap loading; see [production deployment and operations](../ops/deploy.md)).

Steadier ways to release:

- **Canary**: release one new-version instance first, shift a small share of traffic to it by Service label selector or by gateway ratio, watch the metrics (TTFT, error rate, token throughput), then continue. Argo Rollouts / Flagger can automate this.
- **Shadow traffic**: copy production requests to the new version without returning its responses to users, purely to compare outputs and performance. Especially useful when changing model versions, since you can directly compare the two versions' generations.
- **`minReadySeconds`**: watch a new Pod for a few more seconds after it becomes ready before counting it, so instances that "crash right after becoming ready" don't count as available replicas.

## Graceful shutdown: don't cut off users' answers {#优雅退出别把用户的回答截断}

```yaml
spec:
  terminationGracePeriodSeconds: 180     # set by the generation time of the longest answer
  containers:
    - name: server
      lifecycle:
        preStop:
          exec:
            command: ["sh", "-c", "sleep 5"]   # give endpoint removal time to propagate
```

The full sequence of stopping a Pod:

1. The Pod is marked Terminating, and **at the same time** two things happen: it is removed from every Service's endpoint list (no new requests arrive), and the `preStop` hook runs;
2. After `preStop` finishes, the containers receive **SIGTERM**;
3. Wait `terminationGracePeriodSeconds`;
4. If they still haven't exited, **SIGKILL**.

Two pitfalls:

- **Endpoint removal is asynchronous**. kube-proxy, Ingress and gateways each need time to update their rules, and new requests still arrive in the meantime. Sleeping a few seconds in `preStop` (or failing readiness first) covers this window.
- **The process must actually handle SIGTERM**. The inference framework must stop accepting new requests, finish the queued ones, free GPU memory, and then exit. If the process ignores SIGTERM, all that's left is waiting for SIGKILL, and answers mid-generation are cut off. vLLM and SGLang both implement graceful shutdown; if you wrap your own service around them, remember to forward signals (the PID 1 pitfall in containers: start with `exec` or add `tini`).

## Scaling: the wrong metric scales for nothing {#扩缩容指标选错就白扩}

The HPA formula is simple; choosing the metric is the hard part:

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

The simulation shows two things: **scaling up always lags a step behind** (it waits for new replicas to become ready, especially slow for inference services), and **startup time decides how long the overload lasts** (6 overloaded samples with 90-second readiness, only 1 with 15 seconds).

So inference services should configure scaling like this:

| Practice | Reason |
| --- | --- |
| Use **pending requests / queue length / concurrency per replica** as the metric, not GPU utilization | in decode, one request can max out utilization, so it can't tell busy from idle |
| Bring Prometheus in with **KEDA** or a custom metrics adapter | the HPA natively understands only CPU/memory and the custom metrics API |
| Scale up aggressively and down conservatively (configured in `behavior`) | scale-ups lag, so start early; scaling down too fast flaps on traffic jitter |
| Keep a **pre-warmed replica pool** or low-priority placeholder Pods | when you really need to scale up, preempt the placeholders and skip scheduling and image pulls |
| Combine with **Cluster Autoscaler / Karpenter** | when Pods can't scale because nodes are short, scale nodes too, adding several more minutes |

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: {name: vllm}
spec:
  scaleTargetRef: {name: vllm}
  minReplicaCount: 2
  maxReplicaCount: 20
  cooldownPeriod: 300                 # cooldown before scaling down, to avoid flapping
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        query: sum(vllm:num_requests_waiting)    # requests waiting in the queue
        threshold: "5"                            # scale up when 5 requests are waiting per replica on average
```

## PodDisruptionBudget: don't let operations take half the cluster at once {#poddisruptionbudget别让运维一次拿走半个集群}

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: {name: vllm}
spec:
  minAvailable: 80%                   # or maxUnavailable: 1
  selector: {matchLabels: {app: vllm}}
```

Draining a node with `kubectl drain`, cluster autoscaler scale-downs and node upgrades, these **voluntary evictions** all ask the PDB first: "if one more is evicted, is the budget still met?" If not, they wait. It cannot stop node crashes, OOMKills or preemption.

For inference services, think about the PDB together with the release strategy: with `maxUnavailable=1` and a PDB of `minAvailable: 100%`, node maintenance gets stuck outright. A common practice is `maxUnavailable: 1` with enough replicas, so operations can always roll through one node at a time.

!!! interview "In an interview"
    When asked how to deploy an inference service: first point out the two special numbers, minutes to start and minutes to stop, then go item by item. Probes: startupProbe is mandatory (otherwise weight loading is judged a failure by liveness, with endless restarts), liveness must be far more lenient than readiness (otherwise healthy instances get restarted under high load, triggering an avalanche), and readiness can serve as backpressure. Releases: maxSurge needs extra GPUs and maxUnavailable sacrifices capacity, so GPU services usually roll slowly or release at off-peak hours, with canaries and shadow traffic being steadier; startup time multiplies directly into release time. Shutdown: endpoint removal is asynchronous, so cover the window with preStop, set terminationGracePeriodSeconds by the longest answer, and make the process actually handle SIGTERM. Scaling: don't use GPU utilization (one request in decode maxes it out), use queue length or concurrency per replica, bring in Prometheus with KEDA, scale up aggressively and down conservatively, together with node scaling and pre-warmed pools. Finally add the PDB: it governs only voluntary evictions and must be designed together with the release strategy.

## Exercises {#练习}

**1. Compute the release window.** A 12-replica service whose new replicas take 150 seconds to become ready, with a 60-second graceful shutdown. With `maxSurge=2, maxUnavailable=1`, roughly how long does the whole release take? And if the cluster can spare only 1 extra GPU?

??? success "Answer"
    Just run this chapter's `rollout()` (plugging in `replicas=12, ready_s=150, drain_s=60`). An intuitive estimate: each "batch" can replace 3 at once (surge 2 + unavailable 1), so 12 replicas take 4 batches, each with a critical path of "the new replica becoming ready in 150 seconds", so about 4 × 150 = 600 seconds at least, plus the overlap of shutdowns.

    With only 1 extra GPU (`maxSurge=1`), parallelism drops to 2, the number of batches doubles, and release time nearly doubles. Better options then: release at off-peak hours, temporarily add a batch of nodes before releasing, or accept `maxUnavailable=2` and sacrifice a little capacity.

**2. Diagnose repeated restarts.** An inference Pod is stuck in CrashLoopBackOff, and `kubectl logs --previous` shows the log stopping at "Loading model weights..." with no error. What is the most likely cause?

??? success "Answer"
    The liveness probe judged failure before the weights finished loading, and the kubelet restarted the container; after restarting it loads from scratch again, round and round. To confirm: `kubectl describe pod` shows `Liveness probe failed` events, and the container's `Last State` is `Terminated` with `Reason: Error` rather than an application error.

    The fix: add a `startupProbe` (`failureThreshold × periodSeconds` larger than the longest load time), and relax liveness's `initialDelaySeconds` / `failureThreshold`. Also check why loading is slow (weights not in the image, pulled from object storage every time, no local cache volume).

**3. Choose a scaling metric.** A service's GPU utilization sits at 95% for long stretches, but p99 TTFT has risen from 200 ms to 3 seconds. What happens if GPU utilization is the HPA metric? What should be used?

??? success "Answer"
    Nothing happens: utilization is already 95%, the HPA thinks "the target is reached", and doesn't scale. But TTFT rose 15×, showing that requests are queueing.

    Use **queueing metrics**: `num_requests_waiting` (both vLLM and SGLang have it), concurrency per replica, or TTFT percentiles directly. Bring the Prometheus query in with KEDA, with the threshold set as "how many waiting per replica on average". Also lower `scaleUp`'s `stabilizationWindowSeconds` and raise `scaleDown`'s.

**4. Design a safe model version change.** You must replace the production v1 model with v2 (different weights, so outputs change); the service has 20 replicas and an SLO of p99 TTFT < 1 second. Write out the release steps.

??? success "Answer"
    (1) **Shadow traffic**: first start 2 v2 replicas and copy production requests to them (without returning responses to users), comparing output differences and performance metrics over enough samples; (2) **canary**: shift 5% of traffic to v2 (routing by ratio at the gateway, or one more Deployment + weights), and watch production metrics for TTFT, error rate and generation quality for 30 minutes; (3) **gradual ramp-up**: 5% → 25% → 50% → 100%, with an observation window at each step, switching back immediately on abnormal metrics; (4) after full rollout, keep v1's ReplicaSet for a while to allow a quick `kubectl rollout undo`; (5) run the whole process with a PDB and an off-peak window, and confirm in advance that there are enough GPUs to run both versions side by side.

## Summary {#小结}

- [x] Inference services' two special numbers: minutes to start and minutes to stop; every default setting must be recomputed with them.
- [x] startupProbe is mandatory; liveness must be lenient (or high load triggers an avalanche); readiness can serve as backpressure.
- [x] The cost of a rolling release is "extra GPUs (maxSurge) or lost capacity (maxUnavailable)", and startup time multiplies directly into release time; canaries and shadow traffic are steadier.
- [x] Graceful shutdown: endpoint removal is asynchronous (cover the window with preStop), the grace period is set by the longest answer, and the process must actually handle SIGTERM.
- [x] Scale on queue length rather than GPU utilization; scale-ups lag, so scale up aggressively and down conservatively, together with node scaling and pre-warmed pools.
- [x] A PDB governs only voluntary evictions and must be designed together with the release strategy.
