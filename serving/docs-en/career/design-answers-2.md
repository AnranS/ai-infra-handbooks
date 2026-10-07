# System design reference answers (part two): long context, speculative decoding, structured output, observability, multi-region

<p class="lead">Continuing from <a href="../design-answers-1/">the previous chapter</a>, here are reference answers to the other 5 system design problems. These lean more toward "the platform": trading latency against cost, coordinating CPU and GPU, scaling and disaster recovery. As before, every estimate runs, the numbers depend on assumptions, and you should state yours clearly in an interview.</p>

## 6. A million-token long-context service {#6-百万-token-长上下文服务}

> Provide up to a million tokens of context for an MLA + MoE model (DeepSeek-V3 scale): document Q&A, codebase analysis. Time to first token must be "acceptable" and decode speed must not drop noticeably.

**Estimate.**

```python
L, ACTIVE, LAYERS = 1_000_000, 37e9, 61                     # 1 million tokens; an MLA + MoE model (37 billion activated parameters, 61 layers)
print(f"KV：MLA 每 token 70 KB → {L * 70e3 / 1e9:.0f} GB；同尺寸的 GQA 稠密模型（FP8，160 KB）→ {L * 160e3 / 1e9:.0f} GB")
linear = 2 * ACTIVE * L                                     # matrix multiplies over the activated parameters
dense = L * L / 2 * 81920 * LAYERS                          # causal attention: L²/2 pairs at 82K FLOPs each (expanded multi-head attention)
index = L * L / 2 * 16384 * LAYERS                          # the DSA indexer scores all past tokens (FP8, counted at double throughput)
sparse = L * 2048 * 278528 * LAYERS                         # each query does absorbed MLA over only 2048 tokens
rate = 64 * 989e12 * 0.5                                    # 64 GPUs at 50% MFU
print(f"稠密 prefill：线性层 {linear:.1e} + 注意力 {dense:.1e} FLOPs，64 卡约 {(linear + dense) / rate:.0f} s")
print(f"稀疏注意力（DSA）：索引器 {index:.1e} + 选中部分 {sparse:.1e} FLOPs，64 卡约 {(linear + index / 2 + sparse) / rate:.0f} s")
print(f"decode 每步读 KV：稠密 {L * 1152 * LAYERS / 1e9:.0f} GB，DSA {(L * 132 + 2048 * 1152) * LAYERS / 1e9:.0f} GB（索引器的键 + 选中的潜向量）")
```

```text title="输出"
KV：MLA 每 token 70 KB → 70 GB；同尺寸的 GQA 稠密模型（FP8，160 KB）→ 160 GB
稠密 prefill：线性层 7.4e+16 + 注意力 2.5e+18 FLOPs，64 卡约 81 s
稀疏注意力（DSA）：索引器 5.0e+17 + 选中部分 3.5e+16 FLOPs，64 卡约 11 s
decode 每步读 KV：稠密 70 GB，DSA 8 GB（索引器的键 + 选中的潜向量）
```

At a million tokens, attention's compute is over 30 times the linear layers': a dense model takes over a minute for the first token even on 64 GPUs, while sparse attention squeezes it to about ten seconds. Each decode step must read 70 GB of KV (an 8-GPU machine's memory bandwidth is about 27 TB/s, a floor of about 2.6 ms), and sparse attention reads far less.

**Architecture.** Long-context requests go to a dedicated resource pool rather than mixing with ordinary requests (otherwise one million-token prefill holds up the whole instance); prefill uses context parallelism (CP) to split the sequence across machines, with ring / Ulysses communication (see the distributed training handbook's [context parallelism](train://model/context/)); KV is offloaded to the memory pool and SSD; **prefix caching is the biggest lever**, since the second question about the same document or codebase gets its prefill almost free; optional lossy measures (KV eviction) are only for accuracy-insensitive scenarios.

**Trade-offs.** A dedicated pool's low utilization vs the interference of co-location; CP's communication overhead vs one machine not being able to compute it; the product trade-off between time to first token and "returning partial results first" (processing the document in a stream).

**Follow-up: a user uploads the same 500-thousand-token document and asks 10 questions; how do you optimize?**

??? success "Approach"
    Compute the document's KV once and store it in the cache pool (memory + SSD, about 35 GB here), so later questions only prefill the question itself, cutting TTFT from tens of seconds to sub-second. Key points: organize the prompt with the document first and the question after (only a prefix can be reused); route requests about the same document to an instance that has it cached or that can read it back quickly; bill by document hash, or set a retention time for the cache.

## 7. Productionizing speculative decoding {#7-投机解码服务化}

> Enable speculative decoding (EAGLE or MTP) on a chat service, requiring lower latency at low load, no impact on throughput at high load, and the ability to monitor its effect.

**Estimate.** With small batches (memory-bound), the tokens advanced per step and the speedup depend on the acceptance rate $\alpha$ and the number of drafts $k$ (with the cost per draft token counted as 5% of a decode step):

```python
C = 0.05                                                    # the cost per draft token (relative to one decode step)
print("接受率   最佳草稿数   每步前进 token   小 batch 时的加速")
for a in (0.6, 0.8, 0.9):
    best = max(range(1, 9), key=lambda k: (1 - a ** (k + 1)) / (1 - a) / (1 + k * C))
    adv = (1 - a ** (best + 1)) / (1 - a)
    print(f"{a:>6}   {best:>10}   {adv:>13.2f}   {adv / (1 + best * C):>14.2f}x")
```

```text title="输出"
接受率   最佳草稿数   每步前进 token   小 batch 时的加速
   0.6            4            2.31             1.92x
   0.8            8            4.33             3.09x
   0.9            8            6.13             4.38x
```

This is the upper bound: with larger batches, verifying $k+1$ tokens is no longer free and the speedup falls off quickly, even going negative ([advanced speculative decoding](../topics/speculative.md#什么时候有效)).

**Architecture.** The draft module (an EAGLE head or MTP layers) is deployed on the same GPU as the target model, sharing KV management; **adjust the number of drafts dynamically by load**: more drafts with small batches, down to 1 or off once the batch exceeds a threshold; measure acceptance rates separately by business domain (code, chat, translation) and don't speculate on traffic with low acceptance; monitor the tokens advanced per step, the distribution of acceptance rates, TPOT percentiles, and a control group with speculation "on / off".

**Trade-offs.** More drafts = a higher ceiling but more waste; tree drafts have higher acceptance but cost more to verify; the draft model must be updated along with the target model (MTP is naturally in sync, while a standalone draft model must be retrained).

**Follow-up: the acceptance rate suddenly dropped; what could be the cause?**

??? success "Approach"
    The traffic mix changed (say a newly launched workload is code generation, where the draft is inaccurate on such text); the target model was updated but the draft was not; sampling parameters changed (acceptance is naturally lower at high temperature); an implementation problem (quantization or a kernel change making the draft and target numerically inconsistent). Break the acceptance rate down by business domain, model version and temperature, and the cause usually appears quickly.

## 8. A structured output and tool calling service {#8-结构化输出与工具调用服务}

> Provide JSON Schema-constrained output and function calling (tool calling), requiring that constrained decoding not add noticeable latency and that tool calls be parsed as early as possible when streaming.

**Estimate.** Constrained decoding must compute a mask of "which tokens are legal" for each request at every step (over a hundred thousand tokens in the vocabulary), which is CPU work:

```python
STEP_MS = 25                                                # the GPU time of one decode step
for per_req_us, name in ((200, "朴素实现：每步遍历词表"), (20, "预编译语法（大部分 token 的掩码提前算好）")):
    for batch in (32, 256):
        cpu = batch * per_req_us / 1000
        verdict = "可以藏在 GPU 计算后面" if cpu < STEP_MS else "超过一步的时间，成为瓶颈"
        print(f"{name}，batch {batch}：CPU 生成掩码 {cpu:.1f} ms / 步（GPU {STEP_MS} ms）→ {verdict}")
```

```text title="输出"
朴素实现：每步遍历词表，batch 32：CPU 生成掩码 6.4 ms / 步（GPU 25 ms）→ 可以藏在 GPU 计算后面
朴素实现：每步遍历词表，batch 256：CPU 生成掩码 51.2 ms / 步（GPU 25 ms）→ 超过一步的时间，成为瓶颈
预编译语法（大部分 token 的掩码提前算好），batch 32：CPU 生成掩码 0.6 ms / 步（GPU 25 ms）→ 可以藏在 GPU 计算后面
预编译语法（大部分 token 的掩码提前算好），batch 256：CPU 生成掩码 5.1 ms / 步（GPU 25 ms）→ 可以藏在 GPU 计算后面
```

**Architecture.** A grammar engine (XGrammar and the like) precompiles the JSON Schema, computing the masks of "context-independent" tokens in advance and handling only the few stack-dependent tokens at runtime; mask computation **overlaps** with the GPU forward pass (while the GPU computes this step, the CPU prepares the next step's mask), with several threads handling different requests in parallel; grammar state is kept per request and supports rollback (pairing with speculative decoding); the streaming parser for tool calls parses incrementally in the model's call format and fires as soon as a complete function name and arguments are recognized; timeouts and retries on format errors are implemented at the gateway. See [structured output and tool calling](../topics/structured-output.md).

**Trade-offs.** Stronger constraints are safer but can "force" the model into unnatural output (lower quality); in some scenarios "validate afterwards + retry" is cheaper than constrained decoding; complex Schemas are slow to compile, so cache the compiled results.

**Follow-up: does constrained decoding affect the model's output quality?**

??? success "Approach"
    It does: the mask changes the sampling distribution, and when the token the model wants is forbidden it can only take the second best, while a strict format (say having to output a certain field first) can interrupt the model's "train of thought". Common mitigations: let a reasoning model think freely first and apply constraints only to the final answer; design the Schema a little looser; compare task metrics with and without constraints, not just the format-correctness rate.

## 9. Observability and autoscaling for inference {#9-推理可观测性与自动扩缩容}

> Design the metric system and autoscaling for a multi-model inference platform: traffic has a clear daily swing with occasional bursts; the SLO must be met while cost comes down.

**Estimate.**

```python
import math

WEIGHTS = 140e9                                             # 70B of BF16 weights
for name, bw in (("对象存储（1 GB/s）", 1e9), ("本地 NVMe（7 GB/s）", 7e9), ("从已运行的实例经 RDMA 拉取（8 × 50 GB/s）", 400e9)):
    print(f"冷启动加载权重：{name} {WEIGHTS / bw:.1f} s")
print("另外还有：进程启动与 CUDA 初始化约 10～20 s，CUDA Graph 录制与 JIT 编译约 10～60 s（可以缓存）")
QPS, LAT, PER_INST = 200, 12, 16.8                          # arrival rate, average latency (s), the maximum rate one instance sustains under the SLO
need = QPS / PER_INST
growth = 0.30 * 2 / 5                                       # traffic rises 30% in 5 minutes, so it rises another 12% during a 2-minute cold start
print(f"Little 定律：平均并发 {QPS * LAT} 个请求；需要 {need:.1f} 个实例；冷启动的 2 分钟里流量还会涨 {growth:.0%}，"
      f"至少预留 {math.ceil(need * growth)} 个热备实例")
```

```text title="输出"
冷启动加载权重：对象存储（1 GB/s） 140.0 s
冷启动加载权重：本地 NVMe（7 GB/s） 20.0 s
冷启动加载权重：从已运行的实例经 RDMA 拉取（8 × 50 GB/s） 0.3 s
另外还有：进程启动与 CUDA 初始化约 10～20 s，CUDA Graph 录制与 JIT 编译约 10～60 s（可以缓存）
Little 定律：平均并发 2400 个请求；需要 11.9 个实例；冷启动的 2 分钟里流量还会涨 12%，至少预留 2 个热备实例
```

**The metric system.** Three layers: **user experience** (P50 / P99 of TTFT and TPOT / ITL, goodput = the share of requests meeting the SLO, error rate); **engine state** (queue length, running requests, KV usage, preemption count, prefix cache hit rate, batch size per step); **resources and cost** (GPU memory bandwidth and SM utilization, cost per million tokens). Bucket by model, tenant and request length, since averages that mix long and short requests are meaningless.

**Scaling.** Trigger on **queue length, KV usage and the trend of TTFT**, not GPU utilization (naturally modest during decode); **scale ahead** by the historical daily curve, relying on hot standby for bursts; scale down slowly (waiting for requests to finish, avoiding flapping); optimize cold starts: weights on local NVMe or pulled from a running instance over RDMA, with CUDA Graphs and compilation results cached.

**Follow-up: P99 TTFT suddenly rose; how do you investigate?**

??? success "Approach"
    First separate "queueing" from "computing slowly": look at queue length and prefill's own duration. Long queues → a traffic spike or fewer instances (failures, failed scale-ups), or uneven routing (one instance turned into a hotspot by cache-aware routing); slow prefill → more long requests (bucket by length), a lower prefix cache hit rate (a model release, a routing change), or more preemption (not enough KV). Then check the change log (releases, configuration, traffic sources). Breaking metrics down by instance, model and length is the prerequisite for locating anything.

## 10. A multi-region, multi-cluster inference platform {#10-跨地域多集群推理平台}

> Deploy the inference service for the same 700B model in several regions: users connect to the nearest, service continues when any one region fails, and cost is kept as low as possible.

**Estimate.**

```python
MODEL, REGIONS, LINK = 700e9, 12, 10e9 / 8                  # a 700B model in FP8; 12 regions; a 10 Gb/s inter-region link
print(f"逐个地域从中心拷贝：{MODEL / LINK * REGIONS / 3600:.1f} 小时；分块接力分发（收到一块就转发给下一个地域，按两倍余量）约 {MODEL / LINK / 3600 * 2:.1f} 小时")
for n in (3, 4, 6):                                         # when any one region fails, the rest must absorb its traffic
    print(f"{n} 个地域均摊流量：每个地域平时最多跑到 {1 - 1 / n:.0%} 的容量，才能在一个地域故障时不过载")
```

```text title="输出"
逐个地域从中心拷贝：1.9 小时；分块接力分发（收到一块就转发给下一个地域，按两倍余量）约 0.3 小时
3 个地域均摊流量：每个地域平时最多跑到 67% 的容量，才能在一个地域故障时不过载
4 个地域均摊流量：每个地域平时最多跑到 75% 的容量，才能在一个地域故障时不过载
6 个地域均摊流量：每个地域平时最多跑到 83% 的容量，才能在一个地域故障时不过载
```

**Architecture.** Global traffic steering (DNS / Anycast + a global load balancer) allocates by latency and each region's remaining capacity; each region is a complete cluster (gateway, inference instances, KV cache pool), and **the KV cache is not shared across regions** (reading back across regions is slower than recomputing); model distribution uses chunked relays or P2P, with new versions rolled out in one region first; active-active disaster recovery plans each region's capacity so that "N−1 still carries all the traffic" (the more regions, the lower the redundancy ratio); on cost, exploit the time difference between regions to shift deferrable offline tasks, and use spot instances for offline batch work.

**Trade-offs.** More regions mean lower latency and a lower redundancy ratio, but more complex operations and model distribution; sticking sessions to one region (high cache hits) vs scheduling across regions by load (balanced but with cache misses); data compliance may require that certain users' requests never leave the country, which limits scheduling freedom.

**Follow-up: a whole region fails; what happens to long conversations in progress?**

??? success "Approach"
    Streaming requests break off and are retried in another region by the client or the global gateway; conversation history is kept in session storage replicated across regions (as text, not KV), and the new region re-prefills the session (a cache miss, so TTFT grows). To avoid an avalanche, global steering must shift traffic at a limited rate and trigger scale-up in the target region; non-interactive offline tasks are paused first to yield capacity to online traffic.

## Summary {#小结}

- [x] Long context: at a million tokens, attention's compute is tens of times the linear layers', handled with sparse attention, context parallelism, a dedicated resource pool and prefix caching; KV capacity relies separately on offloading.
- [x] Speculative decoding: with an acceptance rate of 0.8 the ceiling is about 3× at small batches, so adjust the number of drafts dynamically by load and monitor acceptance by business domain.
- [x] Structured output: mask computation happens on the CPU, so precompile the grammar and overlap it with the GPU forward pass, or it becomes the bottleneck at large batches.
- [x] Observability and scaling: build metrics in three layers (experience, engine, resources), scale on queueing and KV usage, and handle cold starts with local caches and hot standby.
- [x] Multi-region: with N regions sharing load evenly, each can run at most (N−1)/N of its capacity; KV isn't shared across regions, and models are distributed by chunked relays.
