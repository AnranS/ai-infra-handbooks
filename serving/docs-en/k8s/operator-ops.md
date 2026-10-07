# Operators, quotas and troubleshooting

<p class="lead">Once a platform reaches a certain scale, "every new model takes a dozen YAML files" is bound to happen, and that is when to solidify the pattern into a custom object and a controller, so the business side only writes `model: Qwen3-0.6B, replicas: 3`. This chapter builds an InferenceService CRD from scratch (created on a real cluster) and writes the controller's reconcile logic as runnable code; then it covers the quotas and priorities multi-tenancy needs; finally it gathers the troubleshooting clues of the previous chapters into one table you can follow.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What are a CRD and an Operator? Why does an inference platform need them?
    2. When writing a controller, how do `spec` and `status` divide the work? Why use the `status` subresource?
    3. How does a controller know "which child objects I created"?
    4. What do ResourceQuota and LimitRange each govern?
    5. When a Pod is `Pending`, `CrashLoopBackOff`, or `Running but 0/1`, what do you look at first in each case?

??? success "Answers (try first, then expand to compare)"
    1. A **CRD** (CustomResourceDefinition) registers a new object type with the API server, after which it can be `kubectl get/apply`-ed like built-in objects; an **Operator** is the controller for that object, translating it into real resources such as Deployments, Services and HPAs. An inference platform needs them because "launching a model" involves a fixed set of patterns (deployment + service + autoscaling + monitoring + gateway registration); distill the pattern into one object, and the business side only fills in a few fields.
    2. `spec` is **the user's desired state**, and `status` is **the actual state the controller writes**. The `status` subresource (`subresources: {status: {}}`) separates their update permissions: the controller updating status can't accidentally change spec, users changing spec won't overwrite status, and it avoids the two sides triggering update storms in each other.
    3. Through **ownerReferences**: when creating child objects, write the custom resource as their owner. Then deleting the custom resource **cascades the deletion** to the children (handled by the garbage collector), and the controller can look up the objects it manages through the owner.
    4. **ResourceQuota** is a namespace-level cap on totals (how much CPU, memory and GPU can be used in all, how many Deployments can be created); **LimitRange** sets defaults and bounds at the level of a single Pod/container (a default when requests aren't written, a cap on how much one container may ask for). Once a ResourceQuota is on, **Pods without requests are rejected outright**, so the two are usually used together.
    5. `Pending`: look at `FailedScheduling` in `describe` (a scheduling problem); `CrashLoopBackOff`: look at `logs --previous` and the container's `Last State` (a startup or probe problem); `Running but 0/1`: look at the readiness probe configuration and what `/health` actually returns (the service isn't ready yet, or the probe is misconfigured).

## Custom resources: launch a model in five lines {#自定义资源让上线一个模型只写五行}

![Figure: Kubernetes' controller pattern: declare the desired state, and controllers keep comparing desired with actual](../assets/figures/k8s-reconcile.svg){.aig-svg}

```yaml title="crd.yaml"
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: inferenceservices.ai.example.com
spec:
  group: ai.example.com
  scope: Namespaced
  names: {plural: inferenceservices, singular: inferenceservice, kind: InferenceService, shortNames: [isvc]}
  versions:
    - name: v1alpha1
      served: true
      storage: true
      subresources: {status: {}}          # split status into its own subresource
      schema:
        openAPIV3Schema:                  # the API server validates user-written objects against this schema
          type: object
          properties:
            spec:
              type: object
              required: [model, replicas]
              properties:
                model: {type: string}
                replicas: {type: integer, minimum: 0}
                gpusPerReplica: {type: integer, default: 1}
            status:
              type: object
              properties:
                readyReplicas: {type: integer}
                phase: {type: string}
      additionalPrinterColumns:           # show a few extra columns in kubectl get
        - {name: Model, type: string, jsonPath: .spec.model}
        - {name: Replicas, type: integer, jsonPath: .spec.replicas}
        - {name: Ready, type: integer, jsonPath: .status.readyReplicas}
        - {name: Phase, type: string, jsonPath: .status.phase}
```

After registration, launching a model on the business side takes only:

```yaml
apiVersion: ai.example.com/v1alpha1
kind: InferenceService
metadata: {name: qwen3}
spec:
  model: Qwen/Qwen3-0.6B
  replicas: 3
  gpusPerReplica: 1
```

```bash
kubectl apply -f crd.yaml
kubectl apply -f isvc.yaml
kubectl get isvc
```

```text title="output (on this machine)"
customresourcedefinition.apiextensions.k8s.io/inferenceservices.ai.example.com created
inferenceservice.ai.example.com/qwen3 created

NAME    MODEL             REPLICAS   READY   PHASE
qwen3   Qwen/Qwen3-0.6B   3
```

`READY` and `PHASE` are empty, because no controller has filled them in yet. A CRD only "defines a kind of object"; the Operator does the real work.

## The controller's reconcile {#控制器的-reconcile}

```python title="isvc_controller.py"
"""一个 InferenceService Operator 的 reconcile 逻辑（与 controller-runtime 的 Reconcile 同构）。

真实 Operator 里这段函数的输入是 API server 的对象，输出是对子对象的增删改；
这里把集群抽象成字典，方便把"该建什么、该改什么、状态怎么写回"这套逻辑单独跑通。
"""


def desired_children(spec):
    """从自定义资源的 spec 推导出应该存在的子对象"""
    name, replicas = spec["name"], spec["replicas"]
    return {
        f"deploy/{name}": {"replicas": replicas, "image": spec["image"],
                           "gpus": spec["gpusPerReplica"]},
        f"svc/{name}": {"selector": name, "port": 8000},
        f"hpa/{name}": {"min": max(1, replicas // 2), "max": replicas * 3,
                        "metric": "num_requests_waiting"},
    }


def reconcile(spec, actual):
    """比较期望与实际，返回要执行的动作列表和新的 status（幂等：只看当前状态）"""
    want = desired_children(spec)
    actions = []
    for key, cfg in want.items():
        if key not in actual:
            actions.append(("create", key, cfg))
        elif any(actual[key].get(k) != v for k, v in cfg.items()):
            actions.append(("update", key, cfg))     # compare only the fields we own; leave the state written by child controllers alone
    for key in actual:                                   # delete surplus child objects (say, when replicas drop to 0)
        if key not in want:
            actions.append(("delete", key, None))
    ready = actual.get(f"deploy/{spec['name']}", {}).get("ready", 0)
    phase = "Ready" if ready >= spec["replicas"] and not actions else \
            "Progressing" if actions or ready < spec["replicas"] else "Unknown"
    return actions, {"readyReplicas": ready, "phase": phase}


def apply(actions, actual):
    """把动作施加到"集群"上，模拟子控制器随后把 Pod 拉起来"""
    for op, key, cfg in actions:
        if op == "delete":
            actual.pop(key, None)
        else:
            actual[key] = dict(cfg)
    for key, obj in actual.items():
        if key.startswith("deploy/"):
            obj["ready"] = obj["replicas"]               # assume every Pod becomes ready
    return actual
```

```python
from isvc_controller import apply, desired_children, reconcile

spec = {"name": "qwen3", "replicas": 3, "image": "vllm/vllm-openai:v0.11.0", "gpusPerReplica": 1}
actual = {}

print("第一次 reconcile（集群里什么都没有）：")
actions, status = reconcile(spec, actual)
for op, key, cfg in actions:
    print(f"  {op:6s} {key}")
print("  status：", status)
actual = apply(actions, actual)

print("\n第二次 reconcile（已经收敛）：")
actions, status = reconcile(spec, actual)
print("  动作：", actions or "无")
print("  status：", status)

print("\n把 replicas 改成 6 之后：")
spec["replicas"] = 6
actions, status = reconcile(spec, actual)
for op, key, cfg in actions:
    print(f"  {op:6s} {key} -> {cfg}")
actual = apply(actions, actual)
print("  再跑一次：", reconcile(spec, actual)[1])

print("\n有人手动删掉了 Service：")
actual.pop("svc/qwen3")
actions, _ = reconcile(spec, actual)
print("  动作：", [(op, key) for op, key, _ in actions], "（自愈）")
```

```text title="output"
第一次 reconcile（集群里什么都没有）：
  create deploy/qwen3
  create svc/qwen3
  create hpa/qwen3
  status： {'readyReplicas': 0, 'phase': 'Progressing'}

第二次 reconcile（已经收敛）：
  动作： 无
  status： {'readyReplicas': 3, 'phase': 'Ready'}

把 replicas 改成 6 之后：
  update deploy/qwen3 -> {'replicas': 6, 'image': 'vllm/vllm-openai:v0.11.0', 'gpus': 1}
  update hpa/qwen3 -> {'min': 3, 'max': 18, 'metric': 'num_requests_waiting'}
  再跑一次： {'readyReplicas': 6, 'phase': 'Ready'}

有人手动删掉了 Service：
  动作： [('create', 'svc/qwen3')] （自愈）
```

This logic is isomorphic to the `Reconcile()` you'd write with controller-runtime (Go) or kopf (Python), and four key points show in the output:

- **Idempotent**: the second reconcile takes no action, and `phase` becomes `Ready`;
- **Compare only the fields you own**: the `ready` written back by the child controller is excluded from the comparison, or the controller falls into a loop of "always thinking it needs to update";
- **Self-healing**: if someone deletes the Service by hand, the next reconcile recreates it;
- **Writing back status**: the controller writes the actual state it observed into `status`, from which users and upper-layer systems judge availability.

A real Operator must also handle several things:

| Matter | How |
| --- | --- |
| Ownership of child objects | write `ownerReferences` at creation, so deleting the custom resource cascades to the children |
| Cleaning up external resources | use a **finalizer**: on deletion, run the cleanup first (deregister gateway routes, release external storage), then remove the finalizer |
| Concurrency conflicts | update with optimistic locking (`resourceVersion`), re-reading and retrying on conflict |
| Rate limiting and retries | the workqueue's exponential backoff; failed objects re-enter the queue after a while |
| Version evolution | multiple CRD versions + a conversion webhook; the version with `storage: true` is the actual storage format |
| Validation and defaults | `required` and `default` in the schema, with validating/mutating webhooks for complex rules |

Existing inference Operators in the ecosystem can be used directly or as references: **KServe** (the standard InferenceService, supporting multiple frameworks, canaries and serverless), **KubeAI**, **llm-d** (for disaggregated architectures), **LeaderWorkerSet** (multi-machine groups), and **Kueue** (queues and quotas). The value of building your own usually lies in "hooking up the company's internal storage, gateways, monitoring and approval flows".

## Multi-tenancy: quotas, priorities and isolation {#多租户配额优先级与隔离}

```yaml
apiVersion: v1
kind: ResourceQuota
metadata: {name: team-a, namespace: default}
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    requests.cpu: "64"
    requests.memory: 256Gi
    count/deployments.apps: "10"
```

```bash
kubectl describe quota team-a
```

```text title="output (on this machine)"
Name:                    team-a
Namespace:               default
Resource                 Used  Hard
--------                 ----  ----
count/deployments.apps   1     10
requests.cpu             150m  64
requests.memory          96Mi  256Gi
requests.nvidia.com/gpu  1     8
```

Exceeding it gets rejected **when the Pod is created**, with the error in the ReplicaSet's events (not on the Deployment, which is an easy place to look in vain):

```text title="output (on this machine)"
Warning  FailedCreate  5s  replicaset-controller  Error creating: pods "big-6dd76597ff-nfbnn" is forbidden: exceeded quota: team-a, requested: requests.cpu=100, used: requests.cpu=150m, limited: requests.cpu=64
```

Supporting mechanisms:

- **LimitRange**: fills in defaults for containers without requests (with a ResourceQuota on, Pods without requests are rejected outright with the error `must specify requests.cpu`), and can also bound individual containers.
- **Priority and preemption**: high priority for online inference, low for offline tasks (see [the scheduler](scheduling.md#抢占与优先级)).
- **Kueue**: better suited to GPUs than ResourceQuota: it queues at the **admission layer**, so when resources are short tasks wait in a queue instead of failing to create, and it supports borrowing and reclaiming between queues.
- **Namespaces + NetworkPolicy**: separate teams and restrict network access across namespaces.

A common trade-off: ResourceQuota is "a hard cap, rejected when exceeded", unfriendly to batch jobs (they fail outright instead of queueing); Kueue is "queueing + quota borrowing", better suited to fair sharing in GPU clusters. In production the two are often stacked: ResourceQuota as the backstop against runaway use, Kueue for day-to-day scheduling.

## Troubleshooting: one table to walk through {#排障一张表走完}

Gather the previous chapters' clues into one path. First locate **which layer things are stuck at**:

<!-- i18n:diagram 6aff8855e7 -->
```bash
kubectl get isvc,deploy,rs,pods -l app=vllm      # top to bottom: find the layer whose numbers don't add up
kubectl describe <the stuck layer>               # Events are at the bottom
kubectl logs <pod> --previous                    # when a container crashes, read the previous instance's logs
kubectl get events --sort-by=.lastTimestamp | tail -30
```

| Symptom | Look first at | Common causes |
| --- | --- | --- |
| The custom resource exists but no Deployment | the Operator's logs | the controller crashed, insufficient RBAC permissions, schema validation failed |
| The Deployment exists but the ReplicaSet can't create Pods | events of `describe rs` | **quota exceeded**, LimitRange requirements, an admission webhook rejection |
| Pod `Pending` | `FailedScheduling` in `describe pod` | not enough GPUs, taint/affinity mismatch, fragmentation, an unbound PVC |
| Pod `ContainerCreating` for a long time | events of `describe pod` | slow image pulls (tens of GB), failed volume mounts, the device plugin not ready |
| Pod `CrashLoopBackOff` | `logs --previous`, `Last State` | wrong arguments, out of GPU memory, wrong weight path, **a liveness probe that's too strict** |
| Pod `Running` but `0/1` | the readiness probe config, what `/health` returns | still loading weights (missing startupProbe), wrong probe path or port |
| Pod `OOMKilled` | `Last State` in `describe`, monitoring | memory limits set for the steady state, forgetting the loading peak, `/dev/shm` too small |
| Pods frequently evicted | node events, `kubectl top node` | node memory pressure, BestEffort QoS, node disk pressure |
| The service is reachable but slow/jittery | metrics (TTFT, queue length), CPU throttling | CPU limits too small and throttled, probes wrongly removing endpoints, uneven routing |
| A rolling release is stuck | `rollout status`, `describe rs` | new Pods can't start (all of the above), blocked by a PDB, not enough resources to surge |
| GPU-related anomalies | `nvidia-smi`, DCGM metrics, node events | a GPU falling off, ECC errors, driver/image mismatch, MIG configuration changes |

Two easily missed checkpoints:

- **Events expire** (kept for 1 hour by default), so after-the-fact troubleshooting relies on the logging system and monitoring; production clusters should collect events too.
- **Look at `status.conditions` in `kubectl get pod -o yaml`**: the four conditions `PodScheduled`, `Initialized`, `ContainersReady` and `Ready` tell you exactly which step things are stuck at.

!!! interview "In an interview"
    When asked about Operators: a CRD registers an object type, and an Operator writes the controller that translates it into Deployments/Services/HPAs; spec is the user's desired state and status the actual state the controller writes back, with the status subresource separating update permissions; child objects belong to their owner through ownerReferences with cascading deletion, and external resources are cleaned up with finalizers; reconcile must be idempotent, compare only the fields it owns, and rate-limit and retry with a workqueue. An inference platform's value is collapsing the dozen YAML files of "launching a model" into five lines. For multi-tenancy, cover ResourceQuota (a hard cap; exceeding it fails creation, with the error in the ReplicaSet's events), LimitRange (defaults), priority and preemption, and Kueue's admission queueing, which suits GPUs better. For troubleshooting, give the path: from the custom resource downward, find the layer whose numbers don't add up → describe for events → logs --previous; and know by heart what to look at first for Pending / CrashLoop / 0-1 READY / OOMKilled.

## Exercises {#练习}

**1. Design a CRD.** You want the business side to launch a model service with LoRA in one line, supporting multiple adapters, scaling on queue length, and registering with the gateway automatically. List the fields `spec` should have and the child objects the controller should create.

??? success "Answer"
    `spec`: `baseModel` (the base model), `adapters` (a list of adapters: name + storage path), `replicas` or `autoscaling: {min, max, targetQueueLength}`, `gpusPerReplica`, `resources` (optional overrides), `gateway: {route, priority}`.

    Child objects: a Deployment (or LWS), a Service, a ScaledObject (KEDA), a ConfigMap (the adapter list), an HTTPRoute / InferenceModel (gateway registration), a ServiceMonitor (metric collection), and a PodDisruptionBudget. In status, write `readyReplicas`, `loadedAdapters`, `endpoint` and `conditions`.

**2. Looking in the wrong place.** A Deployment's `AVAILABLE` stays at 0, and `kubectl describe deploy` shows only one `ScalingReplicaSet` event. Where should you look next?

??? success "Answer"
    At the **ReplicaSet**: `kubectl describe rs -l app=<name>`. Events for failed Pod creation (quota, LimitRange, admission webhooks, PSA security policies) hang on the ReplicaSet; the Deployment only says "I told the RS to scale up".

    If the ReplicaSet shows the Pods were created, go further down to the Pods' events and logs. This path of "checking layer by layer, top to bottom, whether the numbers add up" is far more effective than jumping straight to Pod logs.

**3. Quota design.** A team has a quota of 16 GPUs and runs online inference (8 GPUs) alongside offline evaluation (which wants the other 8, but must give them up at peak). How do you configure it?

??? success "Answer"
    (1) A ResourceQuota with `requests.nvidia.com/gpu: "16"` as the hard cap; (2) a high-priority `PriorityClass` with preemption for the online service, and low priority for offline evaluation; (3) manage offline tasks with Kueue queues, with the quota set to "borrowable": when online scales up at peak it preempts offline Pods, and offline tasks automatically return to the queue to wait; (4) implement checkpoints and idempotency in offline tasks so they can resume after preemption; (5) give the online service a PDB so maintenance never takes too much of it at once.

**4. Full troubleshooting.** A newly launched model service shows all 3 Pods as `Running 1/1` in `kubectl get pods`, but the gateway returns 503. Give the order of investigation.

??? success "Answer"
    The Pods are all healthy, so the problem is outside them: (1) **the Service's endpoints**: `kubectl get endpoints <svc>`; if empty, the label selector doesn't match (the Service's `selector` and the Pods' labels disagree) or readiness hasn't passed; (2) **ports**: whether the Service's `targetPort` matches the port the container actually listens on, and whether the container listens on `0.0.0.0` rather than `127.0.0.1`; (3) **gateway configuration**: whether the HTTPRoute/Ingress points at the right Service name and namespace, and whether a path rewrite rule mangles `/v1/...`; (4) **verify with a direct connection from inside the cluster**: `kubectl run -it --rm curl --image=curlimages/curl -- curl http://<svc>:8000/health`; if that works, the problem is in the gateway layer; (5) **gateway logs**: see which backend it chose and what it returned.

## Summary {#小结}

- [x] A CRD defines an object, and an Operator writes the controller that translates it into real resources; inference platforms use them to collapse "launching a model" into a few lines.
- [x] spec is desired, status is actual, separated by the status subresource; child objects cascade-delete through ownerReferences, and external cleanup relies on finalizers.
- [x] reconcile must be idempotent and compare only the fields it owns, or it falls into an update loop.
- [x] ResourceQuota is a hard cap (with errors in the ReplicaSet's events), LimitRange fills in defaults, and GPU clusters are better served by Kueue's admission queueing.
- [x] The troubleshooting path: top to bottom, find the layer whose numbers don't add up → describe for events → logs --previous; Pending / CrashLoop / 0-1 READY / OOMKilled each have a fixed first place to look.
