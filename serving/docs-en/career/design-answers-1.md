# System design reference answers (part one): MoE deployment, a KV cache pool, a gateway, multi-LoRA, RL rollout

<p class="lead">The <a href="../system-design/">system design problems</a> chapter gave the answering framework and one complete example. This chapter and the next give reference answers to the sprint plan's 10 system design problems, each organized as "clarify → estimate → architecture → trade-offs → follow-ups", with every estimate a runnable script. The numbers depend on assumptions, so state yours in an interview; what matters more is the reasoning and the reasons for the trade-offs, not matching these numbers exactly.</p>

## 1. An online inference service for a 600B-class MoE model {#1-600b-级-moe-模型的在线推理服务}

> Design an online service for an MoE model at DeepSeek-V3's scale (671B total parameters, 37B activated, MLA): peak 500 requests per second, average input 3000 tokens and output 800 tokens, with a prefix cache hit rate of about 50%; TTFT P99 < 2 s, TPOT < 50 ms.

**Clarify.** The hardware (H800 / H100, the inter-node network), whether FP8 is allowed (the model is natively FP8, so yes), the daily traffic swing, whether there are very long contexts.

**Estimate.** Use the per-GPU capacity derived from the published data in the [case study](../moe/case-study.md) chapter (about 4000 prefill tokens actually computed per GPU per second, about 1850 output tokens per GPU per second):

```python
import math

QPS, ISL, OSL, HIT = 500, 3000, 800, 0.5        # peak requests/s, average input and output tokens, prefix cache hit rate
PREFILL_GPU, DECODE_GPU = 4000, 1850             # prefill tokens actually computed per GPU per second, output tokens per GPU per second (derived from the published system)
P_UNIT, D_UNIT = 32, 144                         # deployment units: 4 nodes for prefill, 18 for decode
p_gpus, d_gpus = QPS * ISL * (1 - HIT) / PREFILL_GPU, QPS * OSL / DECODE_GPU
p_units, d_units = math.ceil(p_gpus * 1.2 / P_UNIT), math.ceil(d_gpus * 1.2 / D_UNIT)   # leave 20% headroom
print(f"prefill 需要 {p_gpus:.0f} 卡 → {p_units} 个单元（{p_units * P_UNIT} 卡）；decode 需要 {d_gpus:.0f} 卡 → {d_units} 个单元（{d_units * D_UNIT} 卡）")
seqs = DECODE_GPU / 20                            # TPOT 50 ms, so 20 tokens per second per request
kv = seqs * (ISL + OSL / 2) * 70272 / 1e9        # MLA: 70 KB per token
print(f"decode 每卡约 {seqs:.0f} 个并发请求，KV {kv:.0f} GB；总计 {(p_units * P_UNIT + d_units * D_UNIT) // 8} 台 8 卡机")
gpus = p_units * P_UNIT + d_units * D_UNIT
print(f"按每卡每小时 $2：每小时 ${gpus * 2:,}，每百万输出 token ${gpus * 2 / (QPS * OSL * 3600) * 1e6:.2f}")
```

```text title="output"
prefill 需要 188 卡 → 8 个单元（256 卡）；decode 需要 216 卡 → 2 个单元（288 卡）
decode 每卡约 92 个并发请求，KV 22 GB；总计 68 台 8 卡机
按每卡每小时 $2：每小时 $1,088，每百万输出 token $0.76
```

**Architecture.** PD disaggregation; prefill units of 4 nodes (DP attention, EP32 for experts, high-throughput all-to-all, two micro-batch overlap, hierarchical EPLB), decode units of 18 nodes (EP144, low-latency all-to-all, 2 routed experts + 1 shared expert per GPU, global EPLB, MTP); a global scheduler doing cache-aware routing and PD pairing; a distributed KV cache pool (memory + SSD) holding multi-turn conversations' prefixes. The details are in [large-scale EP deployment](../moe/ep-deploy.md) and [NVSHMEM and DeepEP](../comm/nvshmem-deepep.md).

**Trade-offs.**

- The decode deployment unit is large (144 GPUs), making scaling coarse-grained: a smaller unit (say EP72) buys flexibility at the cost of fewer tokens per expert and lower efficiency;
- MTP can slow things down when communication-bound ([MTP and sparse attention](../moe/mtp-sparse.md)), so switch it by load;
- Decode units are underused in traffic troughs: at night some units can be handed to offline tasks or RL rollout.

**Follow-up: what if one decode node fails?**

??? success "Approach"
    With one node in an EP group failing, the whole 144-GPU unit cannot complete its all-to-all. The approach: failure isolation at unit granularity, with the gateway shifting traffic to other units, and requests mid-generation on the failed unit continued on another unit from "the prefix already generated" (its KV can be fetched from the cache pool or recomputed); once a spare node replaces it, weights are reloaded, warmed up and rejoined. So deployments need at least N+1 units, which is one source of the 20% headroom.

## 2. A KV-Cache-centric multi-tier cache pool {#2-以-kv-cache-为中心的多级缓存池}

> Design a KV cache pool for a multi-turn chat + agent platform (a 70B model, 16 machines of 8 GPUs), aiming to lower TTFT and prefill compute: sessions average 8K of context, and users take a few to tens of minutes between turns.

**Estimate.** How many sessions each tier holds, how long reading back takes, and how long recomputing takes:

```python
KV_TOKEN = 160e3                                 # 70B (GQA, FP8 KV) at 160 KB per token
NODES, CTX = 16, 8192                            # 16 machines of 8 GPUs; the average context per session
TIERS = [("GPU 显存（每卡留 40 GB 给 KV）", 8 * 40e9, None),     # (tier, capacity per node, bandwidth reading back into one machine)
         ("内存池（每台 1.5 TB，本机走 PCIe、远端走 RDMA）", 1.5e12, 8 * 50e9),
         ("SSD 池（每台 8 块，共 60 TB）", 60e12, 8 * 6e9)]
session = CTX * KV_TOKEN
recompute = 2 * 70.6e9 * CTX / (8 * 989e12 * 0.5)                # re-prefill on 8 GPUs at 50% MFU
print(f"一个会话的 KV {session / 1e9:.1f} GB；重算需要 {recompute * 1e3:.0f} ms")
for name, cap, bw in TIERS:
    n = NODES * cap / session
    load = f"，读回约 {session / bw * 1e3:.0f} ms" if bw else ""
    print(f"{name}：全集群可存约 {n:,.0f} 个会话{load}")
```

```text title="output"
一个会话的 KV 1.3 GB；重算需要 292 ms
GPU 显存（每卡留 40 GB 给 KV）：全集群可存约 3,906 个会话
内存池（每台 1.5 TB，本机走 PCIe、远端走 RDMA）：全集群可存约 18,311 个会话，读回约 3 ms
SSD 池（每台 8 块，共 60 TB）：全集群可存约 732,422 个会话，读回约 27 ms
```

GPU memory holds only a few thousand active sessions, so most multi-turn conversations with gaps of minutes get pushed out of it; the memory pool holds 5 times more and SSD another 40 times, and reading back from any tier is more than 10 times faster than recomputing.

**Architecture.**

<!-- i18n:diagram cd287b1751 -->
```text
inference instance (GPU KV + local CPU cache) ──write-through / write-back on eviction──▶ distributed KV pool (metadata service + each node's memory) ──▶ SSD pool
        ▲                                                   │
        └──────── on a hit: read back layer by layer (RDMA / PCIe, overlapped with compute) ◀──┘
global key: a prefix-chained hash (mixing in the model version, LoRA and tenant); the router prefers instances that already cached the prefix
```

**Trade-offs.** Write-through (written once computed, with fast eviction and easy sharing but lots of writes) versus write-back (written on eviction, with fewer writes); block granularity (larger blocks mean less metadata and coarser hits); consistency requirements are low (lost data can be recomputed), so no strongly consistent replication is needed; eviction should be ordered by "the time a read-back saves × the probability of being accessed again", not plain LRU. See [KV transfer engines and distributed KV storage](../comm/kv-storage.md).

**Follow-up: what happens to the cache after a model upgrade?**

??? success "Approach"
    KV depends on the weights, so the model version must be mixed into the block keys' seed, and the two versions' blocks then naturally never hit each other; the old version's blocks disappear through eviction, or are deleted in bulk after the switch completes. During the gradual rollout both versions coexist, so plan cache capacity for two, or give the new version its own quota.

## 3. Global scheduling and a cache-aware inference gateway {#3-全局调度与缓存感知的推理网关}

> Design an inference gateway: 200 backend instances, peak 5000 requests per second, requiring cache-aware routing, load balancing, failover and gradual rollout.

**Estimate.**

```python
QPS, INSTANCES, PROMPT, BLOCK = 5000, 200, 4096, 64      # the gateway's peak, the number of backend instances, average prompt length, block size
KV_TOKENS = 2_000_000                                     # tokens each instance can cache
blocks = PROMPT // BLOCK
per_instance = [blocks * INSTANCES, blocks]              # option one: query instance by instance; option two: a global prefix tree whose nodes record the set of instances
print(f"每个请求 {blocks} 个块；逐实例查询每秒 {QPS * per_instance[0] / 1e6:.0f}M 次查找，全局前缀树每秒 {QPS * per_instance[1] / 1e3:.0f}K 次")
nodes = INSTANCES * KV_TOKENS // BLOCK
print(f"索引规模：最多 {nodes / 1e6:.2f}M 个块节点，每个约 64 字节 → {nodes * 64 / 2**20:.0f} MiB，单机内存放得下")
print(f"一个实例故障：健康检查每秒一次、连续 3 次失败才摘除，这 3 秒里发往它的约 {QPS / INSTANCES * 3:.0f} 个请求需要重试；"
      f"用被动检测（请求失败立即标记）可以缩短到几百毫秒")
```

```text title="output"
每个请求 64 个块；逐实例查询每秒 64M 次查找，全局前缀树每秒 320K 次
索引规模：最多 6.25M 个块节点，每个约 64 字节 → 381 MiB，单机内存放得下
一个实例故障：健康检查每秒一次、连续 3 次失败才摘除，这 3 秒里发往它的约 75 个请求需要重试；用被动检测（请求失败立即标记）可以缩短到几百毫秒
```

**Architecture.** Stateless gateway processes in several replicas (behind an L4 load balancer), sharing one routing index (updated from the KV events instances publish, or with each gateway keeping its own approximate index); the routing cost = queueing time + prefill time for the part that missed, with early rejection under overload ([global scheduling](../frontier/disagg-sched.md)); active + passive health checks, circuit breaking, and retries with idempotency keys; gradual rollout splitting traffic by weight, with cache indexes isolated by model version.

**Trade-offs.** Keeping the index in the gateway's memory (fast, inconsistent across replicas) versus in a separate service (consistent, one more hop); the weights of session affinity versus load balancing; when a streaming request fails midway, whether to error out and let the client retry, or have the gateway continue from the content already generated (better experience, but sampling continuity must be preserved).

**Follow-up: what if the gateway becomes the bottleneck?**

??? success "Approach"
    The gateway only routes and forwards, and by the estimate above one core handles tens of thousands of routing decisions per second, so the bottleneck is usually SSE connections and TLS. Scale gateway replicas horizontally; keep the index approximately consistent with "instances publish events → each gateway subscribes", and the occasional routing mistake only costs some extra prefill. Hot prefixes (say one shared system prompt) need a cap on the share one instance takes, so cache-aware routing doesn't pile traffic onto a single instance.

## 4. A multi-tenant LoRA service {#4-多租户-lora-服务}

> An 8B base model serving thousands of LoRA adapters (rank 16), with requests from different tenants arriving mixed; it must load quickly, keep tenants from interfering, and bill per tenant.

**Estimate.**

```python
LAYERS, D, KV, FF, R = 32, 4096, 1024, 14336, 16          # an 8B-class base (GQA), LoRA rank 16, applied to all 7 projections
shapes = [(D, D), (D, KV), (D, KV), (D, D), (D, FF), (D, FF), (FF, D)]   # q k v o gate up down
params = LAYERS * R * sum(i + o for i, o in shapes)
size = params * 2                                          # bf16
print(f"一个适配器 {params / 1e6:.1f}M 参数、{size / 2**20:.0f} MiB；1000 个共 {1000 * size / 2**30:.0f} GiB")
print(f"GPU 上留 8 GiB 给适配器：可常驻 {8 * 2**30 // size} 个；从 CPU 内存经 PCIe（25 GB/s）换入一个 {size / 25e9 * 1e3:.1f} ms")
base = 2 * 8.0e9
print(f"每 token 的计算：基座 {base / 1e9:.0f} GFLOPs，LoRA 额外 {2 * params / 1e9:.2f} GFLOPs（{2 * params / base:.1%}）")
for distinct in (1, 16, 64):                               # how many distinct adapters are in one batch
    print(f"batch 256 个请求用到 {distinct} 个适配器：LoRA 部分要读 {distinct * size / 2**30:.2f} GiB 权重，"
          f"约是读一遍基座（16 GB）的 {distinct * size / 16e9:.0%}")
```

```text title="output"
一个适配器 41.9M 参数、80 MiB；1000 个共 78 GiB
GPU 上留 8 GiB 给适配器：可常驻 102 个；从 CPU 内存经 PCIe（25 GB/s）换入一个 3.4 ms
每 token 的计算：基座 16 GFLOPs，LoRA 额外 0.08 GFLOPs（0.5%）
batch 256 个请求用到 1 个适配器：LoRA 部分要读 0.08 GiB 权重，约是读一遍基座（16 GB）的 1%
batch 256 个请求用到 16 个适配器：LoRA 部分要读 1.25 GiB 权重，约是读一遍基座（16 GB）的 8%
batch 256 个请求用到 64 个适配器：LoRA 部分要读 5.00 GiB 权重，约是读一遍基座（16 GB）的 34%
```

LoRA's compute is nearly negligible; the cost is in **weight reads**: decode is memory-bound, and the more distinct adapters in one batch, the more LoRA weights must be read.

**Architecture.** The base weights are shared; adapters live in three tiers (about 100 popular ones resident on the GPU, all of them in host memory, object storage for persistence), swapped in and out by LRU; batches mix several adapters and compute in one go with segmented GEMM (Punica's BGMV / SGMV, S-LoRA's unified paged memory); the router is affine by adapter so one adapter's requests concentrate on a few instances, lowering the number of distinct adapters per batch; rate limiting and billing per tenant (by tokens, since LoRA's extra cost is small).

**Trade-offs.** The trade-off between how many adapters an instance can serve at once and batch efficiency; higher ranks cost more (linearly); the prefix cache's key must include the adapter ID (LoRA changes K and V).

**Follow-up: what if the first request for an unpopular adapter has very high latency?**

??? success "Approach"
    Swapping an adapter in takes only a few milliseconds (per the estimate above); what is really slow is "downloading from object storage" (hundreds of milliseconds to seconds) and "queueing because its instance's batch is full". The approach: keep all adapters in each machine's memory or local disk in advance; route unpopular adapters to lightly loaded instances; serve the first request on a dedicated "cold start" instance while the adapter is pre-warmed onto its affine instance in the background.

## 5. An RL rollout system {#5-rl-rollout-系统}

> Design a rollout system for RL training of a reasoning model: a 32B policy model, 128 questions × 16 answers per step, average output 8K tokens, about 5 minutes per training step; rollout must not slow training down.

**Estimate.**

```python
PROMPTS, G, AVG_OUT, TRAIN_S = 128, 16, 8000, 300         # 128 questions × 16 answers per step, average output 8K tokens; 5 minutes per training step
TPUT = 2500                                                # a 32B policy model, tokens generated per H100 per second at large batch
tokens = PROMPTS * G * AVG_OUT
for name, util in (("同步（长尾，利用率约 30%）", 0.3), ("全异步（利用率约 90%）", 0.9)):
    gpus = tokens / (TPUT * util * TRAIN_S)                # to keep rollout in step with training
    print(f"{name}：每步 {tokens / 1e6:.1f}M 个输出 token，需要约 {gpus:.0f} 张卡做 rollout")
w = 32e9 * 1                                               # FP8 weights
print(f"权重同步：{w / 1e9:.0f} GB，按 8 卡并行 + 实例间接力约 {w / (8 * 50e9):.2f} s，远小于一步的 {TRAIN_S} s")
```

```text title="output"
同步（长尾，利用率约 30%）：每步 16.4M 个输出 token，需要约 73 张卡做 rollout
全异步（利用率约 90%）：每步 16.4M 个输出 token，需要约 24 张卡做 rollout
权重同步：32 GB，按 8 卡并行 + 实例间接力约 0.08 s，远小于一步的 300 s
```

**Architecture.** Training and inference deployed separately (or co-located at small scale); the inference pool generates continuously with continuous batching, pipelined fully asynchronously or one step asynchronously; the scoring service (rules, sandboxes, reward models) scales independently; weight sync sends point-to-point by the partition mapping; the inference side returns per-token logprobs and (for MoE) routing results. The details are in [RL inference systems](../frontier/rl-async.md).

**Trade-offs.** Synchronous is simple and unbiased but wastes 2/3 of the inference compute; fully asynchronous is efficient, but staleness concentrates on long answers, calling for importance correction, length-balanced batching and a cap on staleness; co-location saves GPUs but requires switching memory.

**Follow-up: what if the scoring service is slow (say code problems that must run tests)?**

??? success "Approach"
    Pipeline scoring with generation: send each answer for scoring as soon as it finishes, without waiting for the whole batch; scale the scoring service independently by CPU / sandbox resources and set timeouts (a timeout counts as a failure, or is dropped and logged); score the 16 answers to one question in parallel. Scoring time also lengthens "sample staleness", so count it toward the staleness cap in a fully asynchronous system.

## Summary {#小结}

- [x] For each problem, first clarify the assumptions that would change the design, then estimate capacity as "demand ÷ per-GPU (per-instance) capacity × headroom", and check the memory, bandwidth and latency constraints.
- [x] A 600B MoE: PD disaggregation + small EP for prefill / large EP for decode, about 70 machines of 8 GPUs to serve 500 requests per second; the KV cache pool: GPU memory holds only a few thousand sessions, while the memory pool and SSD extend it 5–200×, with reading back more than 10× faster than recomputing.
- [x] The gateway: a global prefix tree cuts routing lookups from 64 million per second to 320 thousand; multi-LoRA: the cost is weight reads rather than compute, so route affine by adapter; RL rollout: full asynchrony cuts the inference GPU count to a third.
