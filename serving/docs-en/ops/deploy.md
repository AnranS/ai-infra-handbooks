# Production deployment and operations: Kubernetes, model loading, elasticity and releases

<p class="lead">The previous chapters cared about "how to make one instance fast"; once in production, more problems lie outside the instance: how GPU nodes are scheduled, how hundreds of GB of weights load quickly, when an instance counts as "ready", how to release a new version without interrupting service, what signals to scale on, and where to look when something goes wrong. This chapter goes through these operational questions in the order "deploy → load → health checks → release → scale → observe", and uses a small model to compare the time and capacity costs of several release strategies.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How do you deploy an inference instance spanning two machines (say TP=8 × PP=2) on Kubernetes?
    2. Why do large-model instances need a separately configured startup probe? What should the readiness probe check?
    3. How do you choose maxUnavailable and maxSurge for a rolling release?
    4. Why doesn't autoscaling look at GPU utilization?
    5. How do you cut cold starts from minutes to tens of seconds?

??? success "Answers (try first, then expand to compare)"
    1. With a workload like LeaderWorkerSet: one instance is a group of Pods (a leader plus several workers), together with gang scheduling, so either both machines are scheduled or neither is; then topology labels and affinity put them in adjacent network positions, with RDMA NICs and a local NVMe weight cache configured.
    2. Large models take minutes to load weights, compile and capture CUDA Graphs, during which the liveness probe would wrongly judge them dead and restart them repeatedly; the startup probe gives them enough time. The readiness probe should send a real small request to confirm generation works, not just check that the port is open.
    3. The number of instances taken down at once (maxUnavailable) must not exceed the spare capacity, or the remaining instances can't carry the traffic; when there's headroom or spare resources, add before removing (maxSurge above 0, maxUnavailable at 0), taking old instances down only after new ones are ready.
    4. An inference service's GPU utilization is almost always high (continuous batching keeps the GPU full), so it doesn't reflect overload; look at queue length, KV usage and the trend of TTFT. Scale up fast, scale down slowly, with a minimum instance count above 0.
    5. A local NVMe weight cache (not pulling from remote), streaming parallel loading, pre-sharded weights (split by TP), pre-quantization, and compilation and CUDA Graph caches cut cold starts from minutes to tens of seconds.

## Deploying on Kubernetes {#在-kubernetes-上部署}

![Figure: the shapes of multi-machine instances on Kubernetes: LeaderWorkerSet and PD disaggregation](../assets/figures/lws-pd.svg){.aig-svg}

!!! info "This section is only an overview"
    Kubernetes itself (core objects and controllers, the scheduler, how GPUs are managed, probes and rolling releases, multi-machine multi-GPU and disaggregated deployment, Operators and troubleshooting) gets six chapters in this book's [Kubernetes and inference platforms](../k8s/basics.md) part, each verified item by item on a local k3s. Here we only list the decisions to make when an inference service lands on a cluster.

| Question | Approach |
| --- | --- |
| GPU resources | the NVIDIA GPU Operator (driver, device plugin, DCGM monitoring); Pods request whole GPUs via `nvidia.com/gpu`; label nodes with the model, NVLink topology and NICs, and schedule with affinity |
| Multi-machine instances | when one instance spans several machines (large TP × PP, large-scale EP), use a workload like LeaderWorkerSet that treats "a group of Pods as one unit": created, restarted and scaled together; pair it with gang scheduling (Kueue, Volcano) so half the Pods never get scheduled and hold GPUs while waiting for the other half |
| Networking | expose inter-machine RDMA NICs to Pods through device plugins or SR-IOV; pin the environment variables of NCCL / transfer engines (NIC, GID) in the Pod spec |
| Storage | keep weights in a node-local NVMe cache (pulled from object storage the first time), shared by multiple instances on the same machine; don't download from remote at every startup |
| Traffic entry | an inference gateway (cache-aware routing; see [global scheduling](../frontier/disagg-sched.md)); the Gateway API inference extension for Kubernetes, llm-d and others can choose backends by prefix cache and load |

## Model loading {#模型加载}

The bulk of a cold start is loading weights. For a 70B model (about 140 GB in BF16), pulling from object storage at 1 GB/s takes over two minutes, a local NVMe about 20 seconds, and pulling from an already running instance over RDMA under a second (computed in [the system design reference answers](../career/design-answers-2.md#9-推理可观测性与自动扩缩容)). Common speedups:

- **safetensors + memory mapping**: read on demand, zero copy, and multiple processes share the page cache;
- **Streaming parallel loading**: copy to the GPU while reading, reading multiple files in parallel threads (vLLM's `--load-format runai_streamer`, and `tensorizer`);
- **Pre-sharding**: save weights pre-split for the target TP degree (vLLM's `sharded_state` format), so each rank reads only its own share;
- **Pre-quantization**: save the quantized weights directly, skipping online quantization after loading;
- **Caching compilation artifacts**: keep the compiled results of CUDA Graphs, `torch.compile`, and JIT kernels such as DeepGEMM in a persistent cache directory, or every startup recompiles for tens of seconds.

## Health checks and graceful shutdown {#探活与优雅退出}

- **The startup probe** must allow enough time (loading + compiling + warming up can take minutes), or Kubernetes kills and restarts the container halfway through loading, looping forever;
- **The readiness probe** can't just look at the process and port: it must send a real small request (generating a few tokens) to confirm the model is loaded, CUDA Graphs are captured and results come out correctly before taking traffic;
- **The liveness probe** must be conservative: an inference engine may not answer health checks for a few seconds while processing long requests, and a false kill costs a cold start;
- **Graceful shutdown**: on the termination signal, first leave the load balancer and stop accepting new requests, wait for in-flight requests to finish (`terminationGracePeriodSeconds` must exceed the longest request), then exit. Long streaming requests can be capped, with the gateway continuing timed-out ones from the content already generated.

## Release strategies {#发布策略}

Inference instances cold-start slowly and each one is expensive, so release strategies must trade off time, capacity and extra GPUs. With 16 instances, traffic at 80% of full capacity, a 3-minute cold start and 2 minutes to drain in-flight requests:

```python
import math

N, LOAD, COLD, DRAIN = 16, 0.8, 180, 120        # 16 instances; traffic at 80% of full capacity; new instances cold-start in 3 minutes; old instances drain in-flight requests in 2 minutes

for name, unavail, surge in (("逐个替换（maxUnavailable=1）", 1, 0), ("一次替换 4 个（maxUnavailable=4）", 4, 0),
                             ("先加后减（maxSurge=2）", 0, 2), ("蓝绿（整套新实例）", 0, N)):
    step = unavail + surge                                         # how many to replace per round
    rounds = math.ceil(N / step)
    minutes = rounds * (COLD + DRAIN) / 60
    low = (N - unavail) / N                                        # minimum capacity during replacement
    extra = surge / N                                              # share of extra GPUs needed
    verdict = "不过载" if low >= LOAD else "过载"
    print(f"{name}：{rounds} 轮、约 {minutes:.0f} 分钟；最低容量 {low:.0%}（流量 {LOAD:.0%}，{verdict}）；额外 GPU {extra:.0%}")
```

```text title="output"
逐个替换（maxUnavailable=1）：16 轮、约 80 分钟；最低容量 94%（流量 80%，不过载）；额外 GPU 0%
一次替换 4 个（maxUnavailable=4）：4 轮、约 20 分钟；最低容量 75%（流量 80%，过载）；额外 GPU 0%
先加后减（maxSurge=2）：8 轮、约 40 分钟；最低容量 100%（流量 80%，不过载）；额外 GPU 12%
蓝绿（整套新实例）：1 轮、约 5 分钟；最低容量 100%（流量 80%，不过载）；额外 GPU 100%
```

- Replacing one at a time saves the most GPUs but is slow (over an hour), and rolling back a problem found mid-release is just as slow;
- Replacing too many at once overloads the rest: with traffic at 80%, no more than 20% of instances can go down at the same time;
- Adding before removing (maxSurge) needs a few extra GPUs, but capacity never drops, the first choice when there are spare GPUs;
- Blue-green is fastest with the simplest rollback, but needs double the GPUs, so it's usually used only at off-peak times or at small scale.

**Model version releases** need extra care: first route a small share of traffic to the new version, compare quality metrics (online evaluation, user feedback) and latency metrics between the old and new versions, then widen gradually; prefix caches and KV cache pools must be isolated by model version (KV depends on the weights); with PD disaggregation, the prefill and decode sides must use the same version and switch in pairs during a release.

## Scaling {#扩缩容}

Scale on **queue length, KV usage and the trend of TTFT**, not GPU utilization: SM utilization in decode is naturally modest, and scaling on it either never scales out or scales erratically. Implement it with KEDA or the HPA reading the metrics the engine exposes: vLLM's `vllm:num_requests_waiting` and `vllm:kv_cache_usage_perc`, SGLang's `sglang:num_queue_reqs` and `sglang:token_usage`. Policies:

- **Scale ahead** by the daily curve, absorbing bursts with hot standby instances (size the standby by the traffic growth within one cold-start time);
- **Scale up fast, scale down slowly**: scaling down waits for a stable period and removes only idle instances;
- **The minimum instance count** is not 0 (large models cold-start too slowly; scaling to 0 amounts to stopping service).

## Observability {#可观测性}

| Layer | Metrics | Use |
| --- | --- | --- |
| User experience | P50 / P99 of TTFT and TPOT / ITL, goodput, error rate, bucketed by model, tenant and prompt length | SLOs and alerting |
| Engine | running / queued requests, KV usage, preemptions, prefix cache hit rate, batch size and duration per step | scaling signals, capacity planning, tuning |
| Resources | GPU memory, power, temperature, NVLink / NIC throughput and errors (DCGM), XID errors | early detection of hardware failures |
| Request level | each request's timeline: queueing, prefill, first token, each stretch of decode, transfers (OpenTelemetry tracing) | locating individual slow requests |

GPU hardware failures (ECC errors, NVLink downgrades, XIDs) are the norm in large clusters: there must be automatic node isolation and replacement, or one bad GPU slows down or brings down its whole TP / EP group.

!!! interview "How to explain it"
    Platform roles often ask "how do you launch a 70B model without interrupting service". Go in order: deployment (multi-machine instances with LeaderWorkerSet + gang scheduling, topology awareness), loading (local NVMe caches, streaming parallel loading, pre-sharding, compilation caches, cutting cold starts from minutes to tens of seconds), health checks (enough time in the startup probe, a real request in the readiness probe, graceful shutdown draining in-flight requests), release (add before removing, never taking down more than the headroom at once, gradual rollout for model versions with isolated caches), and scaling (on queueing and KV usage, not GPU utilization; scale up fast and down slowly, with a minimum instance count above 0).

## Exercises {#练习}

**1. Why should the readiness probe send a real request?** The process is already listening on its port; what could still be wrong?

??? success "Answer"
    An available port only shows the HTTP server is up: the model may still be loading, CUDA Graphs may still be capturing, one GPU may have failed to initialize (in a multi-GPU instance, if one rank dies the others hang on the first collective), or the wrong weight version or quantization config may have been loaded, producing garbled output. Sending a fixed small request and checking the output (for example, that the greedy output for a fixed prompt matches expectations) catches these problems and avoids sending traffic to instances that "look alive but are actually unusable".

**2. The prefix cache during a release.** While rolling out a new model version, old and new instances coexist. How should the router handle the prefix cache index?

??? success "Approach"
    Separate the index by model version: a prefix cached on old-version instances is invalid for the new version. When routing, first decide by the rollout ratio which version a request goes to, then do cache-aware routing among that version's instances; a session should preferably finish within one version (session stickiness), or switching to the new version midway means re-prefilling the entire context. When the rollout ends, the old version's index is deleted along with its instances.

## Summary {#小结}

- [x] On Kubernetes: the GPU Operator, topology labels and affinity; multi-machine instances with LeaderWorkerSet + gang scheduling; RDMA NICs and a local NVMe weight cache.
- [x] Cold starts are shortened by local caches, streaming parallel loading, pre-sharding, pre-quantization and compilation caches.
- [x] Give the startup probe enough time, have the readiness probe send real requests, and drain in-flight requests on graceful shutdown.
- [x] Releases: never take down more instances at once than the spare capacity; with headroom, add before removing; roll out model versions gradually with caches isolated by version.
- [x] Scale on queueing, KV usage and TTFT trends, fast up and slow down; observability has four layers (experience, engine, resources, requests), and hardware failures must be isolated automatically.
