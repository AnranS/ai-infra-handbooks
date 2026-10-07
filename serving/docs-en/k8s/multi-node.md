# Multi-machine multi-GPU and disaggregated deployment

<p class="lead">A 671B MoE model needs 16 GPUs across two machines, and the assumption "one replica = one Pod" no longer holds: a replica is a group of Pods that must exist together, become ready together and restart together. One step further, PD disaggregation turns prefill and decode into two independently scaled pools, with KV transferred between them. This chapter explains the mechanisms for expressing these shapes on Kubernetes (LeaderWorkerSet, member discovery through Headless Services, xPyD ratios and routing), and the standard taking shape in the Gateway API Inference Extension.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why can't a multi-machine multi-GPU inference instance use an ordinary Deployment?
    2. What problem does LeaderWorkerSet solve? What is the unit of scaling?
    3. When one worker in a group dies, why does the whole group usually restart?
    4. How is PD disaggregation expressed on Kubernetes? How is the ratio set?
    5. How does an inference gateway differ from an ordinary L7 load balancer?

??? success "Answers (try first, then expand to compare)"
    1. A Deployment's replicas are Pods that are **independent of each other and can be added or removed individually**, while the members of a multi-machine instance must know each other's addresses, build a communication group together, and start and stop together. Expressed with a Deployment, you get problems like "scaling up produces half an instance" and "one member restarts and the communication group breaks".
    2. **LeaderWorkerSet (LWS)** packages "1 leader + N-1 workers" as one replica unit: created together, numbered by group, with stable domain names within the group, supporting whole-group restarts, and scaled by group. It is a CRD a Kubernetes SIG built specifically for multi-machine inference, and multi-machine deployments of both vLLM and SGLang recommend it.
    3. Because the members of a collective are fixed at startup: when one process disappears, the rest hang on all-reduce until a timeout; even if that one is restarted, it cannot rejoin the communication group that was already built. So the policy is usually `RecreateGroupOnHostFailure`, restarting the whole group, at the cost of one failure affecting N Pods.
    4. Two independent workloads (each a Deployment or an LWS) plus a global scheduler/gateway: a prefill pool, a decode pool, and routing that "sends the request to P first, then hands the KV to D". The ratio (xPyD) is set by the two sides' processing capacity; this chapter's model gives the ideal ratio as input/output lengths vary, and production adjusts it dynamically by measured queue lengths.
    5. An ordinary L7 load balancer round-robins by connections or requests; an inference gateway must understand **the cost of a request** (input length, estimated output length) and **the state of instances** (queue depth, KV cache hits, the batch currently running), so it needs model-aware routing. The Gateway API Inference Extension (InferencePool / InferenceModel) standardizes these semantics.

## Multi-machine instances: LeaderWorkerSet {#多机实例leaderworkerset}

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata: {name: vllm}
spec:
  replicas: 2                          # two replicas = two groups
  leaderWorkerTemplate:
    size: 4                            # 4 Pods per group (1 leader + 3 workers)
    restartPolicy: RecreateGroupOnHostFailure    # if one member dies, the whole group restarts
    leaderTemplate:
      spec:
        containers:
          - name: leader
            image: vllm/vllm-openai:v0.11.0
            command: ["sh", "-c", "ray start --head --block & vllm serve $MODEL --tensor-parallel-size 8 --pipeline-parallel-size 4"]
            resources: {limits: {nvidia.com/gpu: 8}}
            ports: [{containerPort: 8000}]
    workerTemplate:
      spec:
        containers:
          - name: worker
            image: vllm/vllm-openai:v0.11.0
            command: ["sh", "-c", "ray start --address=$LWS_LEADER_ADDRESS:6379 --block"]
            resources: {limits: {nvidia.com/gpu: 8}}
```

```python title="lws.py"
"""多机多卡的副本组（LeaderWorkerSet 的模型）与 PD 分离的配比。

一个"副本"不再是一个 Pod，而是"1 个 leader + (size-1) 个 worker"的一组 Pod：
一起创建、一起就绪、一起重启、一起被扩缩容。
"""


def group_pods(name, replicas, size):
    """返回每个副本组的 Pod 名字：<name>-<组号> 是 leader，<name>-<组号>-<序号> 是 worker"""
    out = []
    for g in range(replicas):
        pods = [f"{name}-{g}"] + [f"{name}-{g}-{i}" for i in range(1, size)]
        out.append(pods)
    return out


def gpus_needed(replicas, size, gpus_per_pod):
    return replicas * size * gpus_per_pod


def restart_blast_radius(size, policy):
    """一个 worker 挂掉时受影响的 Pod 数：整组重启 vs 只重启这一个"""
    return size if policy == "RecreateGroupOnHostFailure" else 1


def pd_ratio(prefill_ms, decode_ms_per_token, out_tokens, prefill_gpus, decode_gpus,
             decode_batch=64, prefill_batch=1):
    """PD 分离的配比：让两边的"每秒能处理多少个请求"匹配。

    prefill 实例受算力限制，一次基本只服务一个请求，每个请求占用 prefill_ms；
    decode 实例受带宽限制，一步同时推进 decode_batch 个请求，所以每个请求的实际占用是
    out_tokens * decode_ms_per_token / decode_batch。两边每秒的处理能力相等时配比最优。
    """
    per_request_prefill = prefill_ms / prefill_batch
    per_request_decode = out_tokens * decode_ms_per_token / decode_batch
    ideal = per_request_decode / per_request_prefill
    actual = decode_gpus / prefill_gpus
    bottleneck = "prefill" if actual > ideal else "decode"
    return round(ideal, 2), round(actual, 2), bottleneck


def kv_transfer_ms(kv_bytes, link_gbs):
    """PD 分离要把 KV 从 prefill 实例搬到 decode 实例"""
    return kv_bytes / (link_gbs * 1e9) * 1000
```

```python
from lws import group_pods, gpus_needed, kv_transfer_ms, pd_ratio, restart_blast_radius

print("LeaderWorkerSet：replicas=2、size=4（每组 1 leader + 3 worker），每个 Pod 8 张卡")
for i, pods in enumerate(group_pods("vllm", replicas=2, size=4)):
    print(f"  第 {i} 组：leader={pods[0]}，worker={pods[1:]}")
print("  总卡数：", gpus_needed(2, 4, 8), "张（扩缩容以'组'为单位，不会扩出半个实例）")

print("\n一个 worker 挂掉时的影响面：")
for policy in ("RecreateGroupOnHostFailure", "Default"):
    print(f"  {policy:28s} 受影响 {restart_blast_radius(4, policy)} 个 Pod")
print("  多机推理必须整组重启：通信组的成员变了，剩下的进程会卡在集合通信上")

print("\nPD 分离的配比：理想的 decode : prefill 实例数之比")
print("（prefill 每 1K 输入 75 ms、一次一个请求；decode 每 token 20 ms、一步推进 64 个请求）")
print("  输入 \\ 输出      100 token    500 token   2000 token")
for in_tokens in (1024, 4096, 16384):
    prefill_ms = in_tokens / 1024 * 75
    row = ""
    for out_tokens in (100, 500, 2000):
        ideal, _, _ = pd_ratio(prefill_ms, 20, out_tokens, 1, 1)
        row += f"{ideal:11.2f}  "
    print(f"  {in_tokens:5d} token  {row}")
print("  输出越长、输入越短，就越需要更多 decode 实例；反过来长上下文的场景 prefill 更吃紧。")
print("  实际配比还要按实测的队列长度动态调整（xPyD 的 x、y 通常在 1:1 到 1:4 之间）。")

print("\nKV 传输的代价（一个 4K 上下文的请求，GQA 8 头、128 KB/token）：")
kv_bytes = 4096 * 128 * 1024
for name, gbs in (("同机 NVLink", 450), ("机间 400G RDMA", 50), ("机间 TCP", 5)):
    print(f"  {name:16s} {kv_transfer_ms(kv_bytes, gbs):7.1f} ms")
print("  这段时间要么和 prefill 重叠（逐层传），要么直接算进 TTFT")
```

```text title="output"
LeaderWorkerSet：replicas=2、size=4（每组 1 leader + 3 worker），每个 Pod 8 张卡
  第 0 组：leader=vllm-0，worker=['vllm-0-1', 'vllm-0-2', 'vllm-0-3']
  第 1 组：leader=vllm-1，worker=['vllm-1-1', 'vllm-1-2', 'vllm-1-3']
  总卡数： 64 张（扩缩容以'组'为单位，不会扩出半个实例）

一个 worker 挂掉时的影响面：
  RecreateGroupOnHostFailure   受影响 4 个 Pod
  Default                      受影响 1 个 Pod
  多机推理必须整组重启：通信组的成员变了，剩下的进程会卡在集合通信上

PD 分离的配比：理想的 decode : prefill 实例数之比
（prefill 每 1K 输入 75 ms、一次一个请求；decode 每 token 20 ms、一步推进 64 个请求）
  输入 \ 输出      100 token    500 token   2000 token
   1024 token         0.42         2.08         8.33  
   4096 token         0.10         0.52         2.08  
  16384 token         0.03         0.13         0.52  
  输出越长、输入越短，就越需要更多 decode 实例；反过来长上下文的场景 prefill 更吃紧。
  实际配比还要按实测的队列长度动态调整（xPyD 的 x、y 通常在 1:1 到 1:4 之间）。

KV 传输的代价（一个 4K 上下文的请求，GQA 8 头、128 KB/token）：
  同机 NVLink            1.2 ms
  机间 400G RDMA        10.7 ms
  机间 TCP             107.4 ms
  这段时间要么和 prefill 重叠（逐层传），要么直接算进 TTFT
```

Key points:

- **Members find each other through stable domain names**: LWS creates a Headless Service, Pods in a group get domain names like `vllm-0.vllm.default.svc.cluster.local`, and the leader's address is injected into workers via `LWS_LEADER_ADDRESS`. Multi-machine inference frameworks (Ray, `torch.distributed`, SGLang's `--dist-init-addr`) all need such a stable rendezvous point.
- **Scaling is by group**: `replicas: 2` means two complete instances, so "scaling up half an instance that holds GPUs while waiting for the other half" never happens. To truly guarantee a group gets its GPUs at the same time, pair it with the previous chapter's gang scheduling (Kueue / Volcano).
- **The blast radius of whole-group restarts**: under `RecreateGroupOnHostFailure`, one worker dying restarts 4 Pods. This is correct (the communication group is broken), but it means **the failure blast radius grows with the parallel scale**, so the availability of multi-machine instances comes from "multiple groups of replicas", not from "self-healing within a group".
- **Align the topology**: a group's Pods should land in the same rack / the same NVLink domain, constrained with `topologySpreadConstraints` or LWS's `subGroupPolicy` and node labels; otherwise all-reduce across switches slows every step.

## How a request finds an instance: Services and gateways {#一个请求怎么找到实例service-与网关}

| Mechanism | Role | Notes for inference |
| --- | --- | --- |
| **ClusterIP Service** | L4 load balancing (iptables/IPVS round-robin) | unbalanced for long connections and streaming responses (see [CS Fundamentals: load balancing](root://cs/net/load-balance/)) |
| **Headless Service** (`clusterIP: None`) | resolves directly to all Pod IPs | multi-machine member discovery, and letting the gateway route on its own |
| **Ingress / Gateway API** | L7 routing, TLS, gradual rollout | make sure streaming responses are supported (no buffering) |
| **InferencePool / InferenceModel** (Gateway API Inference Extension) | abstracts "a pool of inference instances" and "model name → backend" as standard objects | the gateway routes by queue depth, KV hits and LoRA affinity |

The nature of inference traffic means a gateway cannot just round-robin:

- **Request costs differ by tens of times**: a 200-token input and a 128K input occupy entirely different amounts;
- **Instance states differ greatly**: some instances still have a dozen requests queued, others just freed up;
- **Cache affinity pays off hugely**: sending the same session / the same system prompt to the same instance saves an entire prefill (see [prefix caching](../engine/prefix-cache.md)).

So the community standardized these semantics as the **Gateway API Inference Extension**: `InferencePool` describes a group of homogeneous inference instances, `InferenceModel` maps a model name (including LoRA adapters) to a pool and declares its priority; on the gateway side, the Endpoint Picker chooses instances by the metrics they report (queue length, KV usage, loaded adapters). llm-d, the vLLM production stack and SGLang Model Gateway are all heading this way.

## The shape of PD disaggregation on Kubernetes {#pd-分离在-kubernetes-上的形状}

PD disaggregation splits a request in two: a prefill instance computes the entire prompt and hands the KV to a decode instance, which generates token by token (the principle is in [PD disaggregation and KV transfer](../distributed/pd-disagg.md)). On Kubernetes it becomes three groups of objects:

<!-- i18n:diagram be423fbcc7 -->
```text
┌────────────────────┐  1. request     ┌────────────────────┐
│ gateway/scheduler  │ ──────────────▶ │ prefill pool       │  (Deployment or LWS, compute-heavy GPUs)
│ (EPP/routing)      │ ◀────────────── │                    │
└────────┬───────────┘  2. KV ready    └────────┬───────────┘
         │ 3. hand off                          │ KV transfer (RDMA / NVLink / shared storage)
         ▼                                      ▼
┌────────────────────┐                 ┌────────────────────┐
│ decode pool        │ ◀────────────── │ KV transfer engine │  (NIXL / Mooncake / LMCache)
└────────────────────┘                 └────────────────────┘
```

The last two sections of the program above give two key accounts:

- **The ratio changes with input and output lengths**: with 1K of input and 2000 tokens of output, the ideal decode:prefill is 8.3 : 1, while with 16K of input and 100 tokens of output it is only 0.03 : 1 (prefill is the bottleneck). So the xPyD ratio is not a constant picked off the top of your head; compute it from the business's length distribution, then adjust it dynamically by measured queue lengths (see [global scheduling in disaggregated architectures](../frontier/disagg-sched.md)).
- **The cost of KV transfer sets the deployment boundary**: a 4K context's KV is about 512 MB, taking 1.2 ms over NVLink within a machine, 10.7 ms over RDMA between machines, and 107 ms over TCP between machines. So PD disaggregation all but requires RDMA (or putting P and D on the same machine split by SM; see [PD multiplexing](../frontier/pd-multiplex.md)), and transfer must go layer by layer, overlapping with prefill compute.

Concrete deployment advice:

- **Scale the two pools separately**: scale the prefill pool by "input token rate" and the decode pool by "concurrent requests / queue length", with two separate HPAs/ScaledObjects.
- **Give heterogeneous GPUs the jobs they're good at**: GPUs with strong compute (H100) do prefill, while GPUs with high bandwidth but weaker compute (H20) do decode; this is exactly what the ridge point table in [the hardware cheat sheet](../career/hardware.md) is for.
- **Get the network working first**: RDMA needs SR-IOV or RoCE CNI plugins, the `IPC_LOCK` capability, and enough hugepages; this is often harder to tune than the inference configuration itself.
- **Failure domains**: when prefill dies the request can be retried (generation hasn't started yet), but when decode dies the answer mid-generation is cut off, so the decode pool has higher availability requirements.

![Figure: the shapes of multi-machine instances on Kubernetes: LeaderWorkerSet and PD disaggregation](../assets/figures/lws-pd.svg){.aig-svg}

## The stateful parts {#有状态的那些东西}

Inference services are "mostly stateless", but a few things are stateful and need separate thought in deployment:

| State | Where it goes | Notes |
| --- | --- | --- |
| Model weights | PVC (ReadOnlyMany) or local NVMe + pre-warming | don't pull from object storage every time; pre-warm onto nodes' local disks with an initContainer or a DaemonSet |
| KV Cache (reused across requests) | instance memory + external KV storage | Mooncake Store / LMCache; reuse across instances needs cache-aware routing |
| LoRA adapters | ConfigMaps are too small; use a PVC or object storage | loaded dynamically, with adapter-affinity routing (see [serving multiple LoRAs](../ops/multi-lora.md)) |
| Sessions and conversation history | external storage (Redis and the like) | don't keep them in the instance, or users lose context whenever the instance changes |
| Metrics and logs | Prometheus / the logging system | Pods get deleted, so ship anything written to disk promptly |

A common misconception is deploying inference services with **StatefulSet**. A StatefulSet provides stable names and storage, at the cost of **ordered startup and shutdown** (rolling updates go one at a time, making releases slow). What an inference service really needs is "managing a group of Pods together", which is what LWS does; unless each instance genuinely needs its own persistent volume, don't use StatefulSet.

!!! interview "In an interview"
    When asked about multi-machine deployment: first say why Deployment doesn't fit: replica members must discover each other, become ready together and restart together, and scaling must be by "group". Then LeaderWorkerSet: 1 leader + N-1 workers packaged as one replica, a Headless Service providing stable domain names as the rendezvous point, and `RecreateGroupOnHostFailure` keeping the communication group consistent, at the cost of a failure blast radius equal to the group size, so availability comes from multiple groups of replicas. Pair it with gang scheduling so the whole group's resources arrive together, and topology constraints to keep a group within one NVLink domain. Then PD disaggregation: two independently scaled pools plus a model-aware gateway, with the ratio computed from input and output lengths (the longer the output, the more decode instances), KV transfer all but requiring RDMA (a 4K context's KV is about 512 MB: 10 ms over RDMA, 100 ms over TCP), and heterogeneous GPUs divided by ridge point. Finally mention that the Gateway API Inference Extension standardized InferencePool / InferenceModel and routing by queue depth and KV hits.

## Exercises {#练习}

**1. Choose the deployment shape.** Three services: (a) a 0.6B small model, single GPU, 20 replicas; (b) a 70B model, TP=4 on one machine; (c) a 671B MoE, TP=8 + PP=2 across two machines. Which object deploys each?

??? success "Answer"
    (a) A Deployment, one GPU per Pod, with an HPA scaling on queue length; (b) a Deployment works too: one Pod requests 4 GPUs and does TP inside the process, with no cross-Pod communication; (c) a LeaderWorkerSet with `size: 2` (8 GPUs per Pod, two Pods forming one instance), with gang scheduling, topology constraints and `RecreateGroupOnHostFailure`.

    The criterion is simple: **a communication group across Pods** → LWS; **multiple GPUs within one Pod** → an ordinary Deployment.

**2. Compute the PD ratio.** The business averages 2K tokens of input and 300 of output. Prefill takes 75 ms per 1K of input; decode takes 20 ms per token, advancing 64 requests per step. What is the ideal P:D ratio? And with prefix caching making 60% of the input hit?

??? success "Answer"
    Prefill takes 150 ms per request; decode takes 300 × 20 / 64 = 93.75 ms per request. The ideal decode:prefill = 93.75 / 150 = 0.63, i.e. about 3 prefill instances to 2 decode.

    With a 60% prefix-cache hit rate, prefill's effective work drops to 60 ms and the ratio becomes 93.75 / 60 = 1.56, i.e. 2 prefill to 3 decode: **the cache hit rate directly changes the ratio**. This is also why xPyD must be adjusted dynamically from measurements rather than hard-coded.

**3. A worker dies.** One LWS group has 4 Pods, and one node fails. Describe what happens under `RecreateGroupOnHostFailure` and under not restarting the whole group, and why inference should choose the former.

??? success "Answer"
    Without a whole-group restart: that one Pod is recreated, but the other 3 processes are still in the old communication group waiting for it; the new process cannot join the established NCCL group, so the 3 old processes hang in collectives until a timeout (minutes), during which this instance is completely unavailable, and it may stay stuck without recovering.

    With a whole-group restart: all 4 Pods are deleted and recreated together, and service resumes after a few minutes (weights must be reloaded). The blast radius is 4 Pods, but **the state is deterministic**. Inference services must choose the latter, relying on multiple groups of replicas + a PDB to guarantee overall capacity.

**4. What should the gateway watch?** You must write routing logic for a pool of 20 instances. List three instance-side metrics you must collect and how each is used.

??? success "Answer"
    (1) **Pending requests / queue depth**: reflects busy versus idle directly, as the main load metric (much more accurate than connection counts); (2) **KV Cache usage**: close to full, this instance is about to start preempting or rejecting, so steer away early; (3) **prefix cache hit information** (which prefixes an instance has cached, or an approximate view kept by the router): used for cache-affinity routing, where a hit saves an entire prefill.

    How to combine them: first pick candidates by cache affinity, then choose the least busy candidate by queue depth, with a load cap to avoid hotspots (consistent hashing with bounded loads; see [CS Fundamentals: consistent hashing](root://cs/dist/hash-shard/)). Also collect TTFT/TPOT percentiles for alerting and the HPA.

## Summary {#小结}

- [x] Use LeaderWorkerSet for communication groups across Pods: a group is created together, becomes ready together and restarts together, and scaling is by group.
- [x] Members rendezvous through a Headless Service's stable domain names; gang scheduling ensures the whole group's resources arrive together; topology constraints keep a group within one NVLink domain.
- [x] The blast radius of a whole-group restart equals the group size, so availability comes from multiple groups of replicas rather than self-healing within a group.
- [x] PD disaggregation = two independently scaled pools + a model-aware gateway; the ratio changes with input/output lengths and cache hit rate, and KV transfer all but requires RDMA.
- [x] Inference gateways must route by request cost and instance state, and the Gateway API Inference Extension is standardizing these semantics.
- [x] Weights, KV, adapters and sessions each have their own home; don't deploy stateless inference services with StatefulSet.
