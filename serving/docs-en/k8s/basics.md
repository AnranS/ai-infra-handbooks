# Kubernetes core objects and the controller pattern

<p class="lead">Almost every inference platform runs on Kubernetes: a model service is a Deployment, a load test is a Job, a multi-machine multi-GPU instance is a group of numbered Pods, and rolling releases, scaling and self-healing all rely on controllers. This chapter first explains the core, "declarative API + controller loop", thoroughly: it explains why nothing happening after `kubectl apply` is normal, why a deleted Pod grows back on its own, and why controllers can restart at any time. Then it goes through the objects an inference service actually uses, and finally gives the first set of troubleshooting commands. This chapter's YAML and output were all run on a real single-node cluster.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How do "declarative" and "imperative" differ? What actually happens after `kubectl apply`?
    2. Why must a controller's reconcile be idempotent?
    3. How are Deployment, ReplicaSet and Pod related? Who moves during a rolling update?
    4. What do multiple containers in one Pod share, and what don't they? What are initContainers and sidecars each for?
    5. How do requests and limits differ? What happens if you only write limits?

??? success "Answers (try first, then expand to compare)"
    1. Imperative is "perform this action" (`docker run`); declarative is "here is what the final state should look like". `kubectl apply` merely writes an object into etcd, and it is done once the API server returns 200; the real work is done by the controllers, which watch object changes, compare "desired state" with "actual state", and pull the two together step by step. So a successful apply does not mean the service is available; look at the actual state in `kubectl get`/`describe`.
    2. Because a controller may restart at any time, and events may be delivered twice or missed (hence the periodic full resync as well). Reconcile decides the next action only from "the current desired and the current actual", not from "what I did last time", so repeated execution does no harm.
    3. A Deployment manages ReplicaSets, and a ReplicaSet manages Pods. Each change to the Pod template makes the Deployment create a new ReplicaSet, then **adjust the replica counts of both the new and the old ReplicaSet together**: the new one goes up, the old one down, at a pace set by `maxSurge` and `maxUnavailable`. A rollback simply raises the old ReplicaSet's replica count again.
    4. Containers in the same Pod share the network namespace (one IP, talking to each other over `localhost`) and can mount the same volumes, but **do not share the file system or the process space** (by default). initContainers run to completion in order before the main containers start, suited to downloading weights and pre-flight checks; sidecars run alongside the main container, suited to log collection, metrics export and proxies.
    5. requests are used for scheduling (deciding which node a Pod can go to) and for CPU weight allocation; limits are runtime caps (exceeding CPU means throttling, exceeding memory means OOMKill). With only limits written, Kubernetes sets requests equal to limits, and the QoS becomes Guaranteed: for GPU inference services this is usually exactly what you want, but it wastes quota for CPU.

## Declarative API: nothing happening after apply is normal {#声明式-apiapply-之后什么都没发生才是正常的}

Every interaction with Kubernetes manipulates **objects**: you write the "desired state" into the API server (backed by etcd), and controllers pull reality toward it.

```yaml title="deploy.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
  labels: {app: vllm}
spec:
  replicas: 3                       # desired state: I want 3 replicas
  selector:
    matchLabels: {app: vllm}        # which Pods this Deployment manages, via a label selector
  template:                         # Pod template: what each replica looks like
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

```text title="output (on this machine)"
deployment.apps/vllm created

NAME                   READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/vllm   0/3     3            0           12s

NAME                        READY   STATUS              RESTARTS   AGE
pod/vllm-6ccd564c7d-chg4r   0/1     ContainerCreating   0          12s
pod/vllm-6ccd564c7d-mnw9g   0/1     ContainerCreating   0          12s
pod/vllm-6ccd564c7d-ssnjp   0/1     ContainerCreating   0          12s
```

`apply` took only a few milliseconds, and all it did was write the object. What follows is a relay of controllers: the Deployment controller creates a ReplicaSet, the ReplicaSet controller creates 3 Pods, the scheduler picks a node for each Pod, and the kubelet on the node pulls the image and starts the containers. **If any link gets stuck, the status in `kubectl get` stays stuck there**, so the first step in troubleshooting is always to look at the actual state and events, not to apply again.

## The controller pattern: the reconcile loop {#控制器模式reconcile-循环}

Every controller does the same thing:

<!-- i18n:diagram b869b30e42 -->
```text
for {
    desired := read the object's spec from the API server
    actual  := read the object's status / related child objects from the API server
    if they differ { take one step to bring them closer }
    wait for the next event or a periodic resync
}
```

Writing this loop out takes only thirty lines:

```python title="reconcile.py"
# the controller pattern: read the "desired state" and the "actual state", compute the difference, and take only the one step that brings them closer
from collections import deque


class Cluster:
    """极简的 API server：只存对象，不做任何决策"""

    def __init__(self):
        self.deployments = {}                  # name -> {"replicas": desired replicas, "image": version}
        self.pods = {}                          # name -> {"owner": ..., "image": ..., "phase": ...}
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
        cluster.create_pod(name, spec["image"])            # too few: add one
    elif len(pods) > spec["replicas"]:
        cluster.delete_pod(sorted(pods)[-1])               # too many: delete one
    elif stale:
        cluster.delete_pod(sorted(stale)[0])               # enough of them but an old version: replace one
    else:
        return False                                        # converged: nothing to do
    return True


def run(cluster, name, max_steps=50):
    """控制循环：反复 reconcile 直到不再产生动作"""
    steps = 0
    while steps < max_steps and reconcile(cluster, name):
        for pod in cluster.pods.values():                  # simulate kubelet: Pending Pods become Running after a while
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

# someone deletes a Pod by hand: the controller adds it back on its own; this is "self-healing"
victim = sorted(c.owned("llm"))[0]
c.delete_pod(victim)
print(f"手动删掉 {victim} 之后：", run(c, "llm"), "步补回", len(c.owned("llm")), "个")
print("\n事件流水（前 12 条）：", c.events[:12])
print("要点：reconcile 只根据当前状态决定下一步，重复调用不会出错（幂等），所以控制器可以随时重启。")
```

```text title="output"
扩到 3 个副本： 3 步 ['llm-1', 'llm-2', 'llm-3']
缩到 1 个副本： 2 步 ['llm-1']
换版本到 v2：   {'llm-4': 'v2', 'llm-5': 'v2'}
手动删掉 llm-4 之后： 1 步补回 2 个

事件流水（前 12 条）： ['create llm-1', 'create llm-2', 'create llm-3', 'delete llm-3', 'delete llm-2', 'create llm-4', 'delete llm-1', 'create llm-5', 'delete llm-4', 'create llm-6']
要点：reconcile 只根据当前状态决定下一步，重复调用不会出错（幂等），所以控制器可以随时重启。
```

Several key conclusions are in the output:

- **Scaling, changing versions and self-healing all use the same loop**. Delete a Pod by hand, and the next reconcile notices there are too few and adds one: that is "self-healing", with no special logic at all.
- **Reconcile must be idempotent**. It looks only at the current state, so repeated calls, controller restarts and duplicated events do no harm. Real Kubernetes also does a periodic **full resync**, specifically to catch missed events.
- **One step at a time**. A controller doesn't change the state all the way in one go; it takes one action, writes back the status, and waits for the next event. This makes the whole system observable, interruptible and rate-limitable.

Real controllers add two more layers: the **informer** (a local cache + watching events, avoiding a full read from the API server every time) and the **workqueue** (deduplication, rate limiting, retry on failure). When writing an Operator, controller-runtime provides both, and you only fill in the `Reconcile()` function.

![Figure: Kubernetes' controller pattern: declare the desired state, and controllers keep comparing desired with actual](../assets/figures/k8s-reconcile.svg){.aig-svg}

## The objects inference services use {#推理服务会用到的对象}

| Object | What it's for | In inference |
| --- | --- | --- |
| **Pod** | a group of containers sharing network and volumes, the smallest unit of scheduling | one inference instance (possibly with logging and metrics sidecars) |
| **Deployment** | a stateless set of replicas, supporting rolling updates and rollback | a single-GPU or single-machine multi-GPU model service |
| **StatefulSet** | a set of replicas with stable names and storage, started and stopped in order | scenarios needing fixed identities (a few kinds of distributed inference) |
| **Job / CronJob** | tasks that end when done | load tests, offline batch inference, accuracy evaluation |
| **Service** | a stable entry point for a group of Pods (ClusterIP / Headless) | L4 load balancing from gateway to instances; Headless for multi-machine member discovery |
| **Ingress / Gateway API** | L7 entry, routing, TLS | the external `/v1/chat/completions` |
| **ConfigMap / Secret** | configuration and credentials | model parameters, routing tables, HuggingFace tokens |
| **PVC / PV** | persistent volumes | model weight caches, the disk tier for KV offloading |
| **HPA / KEDA** | scaling by metrics | scaling up by queue length, TTFT, GPU utilization |
| **PodDisruptionBudget** | limits concurrent voluntary evictions | keeping at least N instances online while upgrading nodes |
| **CRD + Operator** | custom objects and controllers | LeaderWorkerSet, Volcano's Queue, your own InferenceService |

Between Deployment and Pod sits the **ReplicaSet**:

```bash
kubectl describe deploy vllm | head -14
```

```text title="output (on this machine)"
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

Change the image once, and the Deployment creates a new ReplicaSet, then the two trade places at the pace of `maxSurge` / `maxUnavailable` (expanded in the next chapter). Rollback (`kubectl rollout undo`) raises the old ReplicaSet's replica count back up; old ReplicaSets are kept for 10 versions by default, which answers "why does the cluster have a pile of ReplicaSets with 0 replicas".

## Inside a Pod: containers, init and probes {#pod-内部容器init-与探针}

The typical structure of an inference Pod:

```yaml
spec:
  initContainers:
    - name: fetch-weights              # first pull the weights from object storage to local disk; the main container starts only after that
      image: rclone/rclone:1.68
      args: ["copy", "s3:models/qwen3-0.6b", "/models/qwen3-0.6b"]
      volumeMounts: [{name: models, mountPath: /models}]
  containers:
    - name: server                     # main container: the inference engine
      image: vllm/vllm-openai:v0.11.0
      volumeMounts: [{name: models, mountPath: /models}]
    - name: metrics-proxy              # sidecar: forward / transform metrics
      image: nginx:1.27-alpine
  volumes:
    - name: models
      emptyDir: {}                     # containers in the same Pod share this volume
```

- **Shared**: the network namespace (one Pod IP, containers talk over `localhost`), mounted volumes, and the lifecycle (when the Pod is deleted, its containers go with it).
- **Not shared**: the file system root and the process space (unless `shareProcessNamespace` is explicitly turned on).
- **initContainers** run in order and exit when done, suited to prerequisite work like "download weights", "check the GPU driver" and "warm the page cache". If one fails it retries forever, and the Pod stays at `Init:0/1`.
- **Sidecars** run alongside the main container. Since 1.29 there are native sidecars (written as initContainers with `restartPolicy: Always`), which start before the main container and stop after it, better suited to logging and proxies than traditional sidecars.

## requests, limits and QoS {#requestslimits-与-qos}

```yaml
resources:
  requests: {cpu: "8", memory: 32Gi, nvidia.com/gpu: 1}
  limits:   {cpu: "16", memory: 64Gi, nvidia.com/gpu: 1}
```

- **requests decide scheduling**: the scheduler judges fit by a node's "allocatable minus what is already taken by requests", regardless of actual use.
- **limits decide runtime behavior**: exceeding CPU means cgroup throttling (not killed, but latency soars; see [CS Fundamentals: containers](root://cs/os/containers/)); exceeding memory means an immediate OOMKill.
- **GPUs must be whole numbers, and requests must equal limits**: `nvidia.com/gpu` is an incompressible extended resource without fractions (sharing a GPU needs MIG or time slicing; see chapter three).
- **Three QoS classes**: requests == limits with both written is `Guaranteed` (least likely to be evicted); only requests is `Burstable`; neither is `BestEffort` (killed first when the node runs short of memory). Inference services should be Guaranteed.

A common pitfall: **setting CPU limits very low**. An inference process's CPU cost is more than "a bit of scheduler logic": tokenization, sampling, HTTP, metrics and PyTorch's thread pools all need CPU; throttling shows up as TTFT jitter and GPU utilization that won't climb, while monitoring shows CPU "not fully used" (because it is being throttled).

## The first set of troubleshooting commands {#排障的第一组命令}

```bash
kubectl get pods -l app=vllm -o wide          # status, restarts, which node, Pod IP
kubectl describe pod <name>                   # Events are at the bottom; 90% of causes are there
kubectl logs <name> --previous                # logs of the previous container instance (most useful after a crash restart)
kubectl get events --sort-by=.lastTimestamp   # the event stream of the whole namespace
kubectl exec -it <name> -- nvidia-smi         # look at the GPU from inside the container
kubectl top pod / kubectl top node            # live resource usage (requires metrics-server)
```

Locate by Pod status:

| Status | Common causes |
| --- | --- |
| `Pending` | no node satisfies the requests (not enough GPUs, taint/toleration mismatch, affinity conflicts); look for `FailedScheduling` in `describe` |
| `ContainerCreating` | slow image pulls, failed volume mounts, the device plugin not ready |
| `CrashLoopBackOff` | the container exits right after starting; look at `logs --previous`: wrong arguments, out of GPU memory, wrong weight path |
| `OOMKilled` (in Last State in `describe`) | memory limits too small; watch the peak while loading weights and `/dev/shm` |
| `Running` but `0/1 READY` | the readiness probe hasn't passed: weights are still loading, or the probe path/timeout is wrong |
| `Terminating` for a long time | a long graceful shutdown (in-flight requests not finished), or the process doesn't respond to SIGTERM |

!!! interview "How to explain it"
    To explain Kubernetes: start with the core, the declarative API plus the controller loop. apply only writes an object; the real work is each controller's reconcile: read desired, read actual, take one step; it must be idempotent because controllers restart and events get duplicated or lost, hence the periodic resync too. From this, explain that self-healing and rolling updates are the same mechanism: the Deployment adjusts the replica counts of the new and old ReplicaSets, at a pace set by maxSurge/maxUnavailable. Then say which objects an inference service maps onto (Deployment, Job, Service, PVC, HPA, PDB, CRD), and the difference between requests and limits: requests decide scheduling, limits decide throttling and OOMKill, GPUs must be whole numbers with requests == limits, and services should be Guaranteed. Finally, the troubleshooting routine: first `get -o wide` for the status, then `describe` for events, `logs --previous` for crash causes, with Pending / CrashLoopBackOff / 0/1 READY corresponding to scheduling, startup and probe problems respectively.

## Exercises {#练习}

**1. apply succeeded but there are no Pods.** `kubectl apply -f deploy.yaml` returns `deployment.apps/vllm created`, but `kubectl get pods` is empty. List three possible causes and the commands to investigate each.

??? success "Answer"
    (1) **The ReplicaSet wasn't created**: `kubectl get rs -l app=vllm` and `kubectl describe deploy vllm` for events; common causes are a `selector` that doesn't match `template.metadata.labels` (the API server rejects it outright), or being blocked by an admission controller (such as an OPA/Kyverno policy); (2) **the Pods were created but in another namespace**: check the `-n` flag and the current context; (3) **insufficient quota**: `kubectl describe quota` and `kubectl get events`; an exceeded ResourceQuota makes the ReplicaSet keep failing to create Pods, with `exceeded quota` in the events.

**2. Why controllers must be idempotent.** Suppose reconcile were written as "remember how many I created last time and only add the difference this time". What happens after the controller restarts?

??? success "Answer"
    After a restart, the in-memory "how many I created last time" is gone; if it defaults to 0, it creates the replicas all over again, doubling the Pods; if the state is stored elsewhere, it may disagree with reality (say someone deleted a Pod by hand). The right way is to **observe the actual state anew** every time (list how many Pods belong to me) and then decide the action, so no matter how many restarts or duplicated events, the result converges to the desired state.

**3. Design the resource spec.** An inference instance needs one H100, peaks at 40 GB of memory (55 GB while loading weights), and uses 4 CPU cores normally, spiking to 12 while batching and tokenizing. Write out `resources` and explain why.

??? success "Answer"
    ```yaml
    resources:
      requests: {cpu: "8", memory: 64Gi, nvidia.com/gpu: 1}
      limits:   {cpu: "16", memory: 64Gi, nvidia.com/gpu: 1}
    ```
    Memory requests = limits = 64 Gi: set by **the loading peak**, not the steady state, with some headroom, or it gets OOMKilled while loading weights; making the two equal puts it in the Guaranteed class, harder to evict. CPU requests take a value above the steady state (8) to guarantee scheduling onto a node with enough free CPU, and limits go to 16 to allow bursts: CPU is compressible, so exceeding it only means throttling. GPUs must be whole numbers with requests and limits equal. Also, don't forget `/dev/shm`: many inference frameworks use shared memory for inter-process communication, which defaults to only 64 MB, so mount a bigger one with `emptyDir: {medium: Memory}`.

**4. Reading ReplicaSets.** `kubectl get rs` shows two ReplicaSets: `vllm-6ccd` desires 3, and `vllm-7f9a` desires 0. What does this mean? To return to `vllm-7f9a`'s version, what is the command?

??? success "Answer"
    It means a rolling update happened: `vllm-6ccd` is the current version, and `vllm-7f9a` is the previous one (its replica count lowered to 0, but the object kept for rollback and history). Roll back with `kubectl rollout undo deploy/vllm` (to the previous version) or `kubectl rollout undo deploy/vllm --to-revision=N`; `kubectl rollout history deploy/vllm` lists the versions. How many historical versions are kept is controlled by `spec.revisionHistoryLimit` (10 by default).

## Summary {#小结}

- [x] Declarative API: apply only writes an object, and controllers' reconcile loops pull reality toward the desired state; so a successful apply ≠ an available service.
- [x] Reconcile looks only at the current state, takes one step at a time, and must be idempotent; informers handle caching, and workqueues handle deduplication and retries.
- [x] Deployment → ReplicaSet → Pod; rolling updates and rollbacks both adjust the replica counts of two ReplicaSets.
- [x] initContainers do prerequisite work (pulling weights), sidecars do side jobs (logs, metrics, proxies), and containers in one Pod share network and volumes.
- [x] requests decide scheduling, limits decide throttling and OOMKill; GPUs must be whole numbers with the two equal; inference services should be Guaranteed, and don't forget `/dev/shm`.
- [x] Troubleshooting order: `get -o wide` → `describe` (look at Events) → `logs --previous`; Pending means scheduling, CrashLoop means logs, 0/1 READY means probes.
