# Large-scale EP deployment: hierarchical EPLB, DP attention load, and two-batch overlap

<p class="lead">The <a href="../../distributed/expert-parallel/">expert parallelism and DP Attention</a> chapter implemented EP + DP Attention on 4 processes and used redundant experts and re-placement to bring the busiest GPU back to the average. At the scale of tens or hundreds of GPUs, three new problems appear: load balancing must cooperate with routing constraints and node topology; DP attention ranks carry different loads, yet MoE's all-to-all synchronizes them at every layer; and how to overlap communication with compute. This chapter answers each with a simulation, and finally looks at why prefill and decode use EP of very different sizes.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What is DeepSeek-V3's "group-limited routing"? How does it relate to EPLB's hierarchical policy?
    2. What scenarios do hierarchical EPLB and global EPLB each suit?
    3. Under DP Attention, why does one rank with especially long requests slow down every rank? How is it mitigated?
    4. When does two-batch overlap pay off most? Why does it barely help with very small batches?
    5. Why is decode's EP size usually much larger than prefill's?

??? success "Answers (try first, then expand to compare)"
    1. Each token first scores by node (group) and picks experts only within the 4 highest-scoring nodes (going to at most 4 nodes). Hierarchical EPLB first assigns expert groups to nodes, then replicates and places experts within each node, guaranteeing that after replication each token still goes to at most 4 nodes, preserving this cap of the routing.
    2. Hierarchical EPLB: prefill with few nodes (small EP), preserving the node limit and cutting cross-node traffic; global EPLB: decode with very large EP, unconstrained by node boundaries, so the load can be balanced more evenly.
    3. MoE's all-to-all needs every rank to participate at every layer, effectively one synchronization per layer: a rank with especially long requests (its attention reads more KV) computes attention slowly, and every rank waits for it. Mitigations: assign requests by total KV rather than by request count, have idle ranks run empty batches alongside, and use PD disaggregation to smooth decode's load.
    4. When compute and communication take about the same time (this chapter estimates 1.45–1.65× faster at a per-GPU batch of 64–128). With a very small batch, compute is almost all weight reads, and after splitting into two micro-batches each re-reads the weights, doubling the weight-read time and canceling the gain from overlap.
    5. In decode each token passes through each expert only once with little compute, so the global batch must be pooled so each expert gets enough tokens; and memory is saved for KV (each GPU holds only a few experts). Prefill has plenty of tokens to begin with, so a small EP already feeds the experts, with fewer nodes and less cross-machine traffic.

## EPLB and routing constraints: hierarchical or global {#eplb-与路由约束分层还是全局}

![Figure: EPLB: pack expert groups onto nodes by load, replicate the hottest experts within each node, then spread them over GPUs](../assets/figures/eplb.svg){.aig-svg}

DeepSeek-V3's 256 experts are divided into 8 groups, and routing takes two steps: first pick 4 groups by the sum of each group's top 2 scores, then pick 8 experts within those 4 groups (**group-limited routing**). If each group's experts all sit on the same node, a token goes to at most 4 nodes, capping cross-node traffic.

The open-source EPLB implementation (DeepSeek's `eplb.py`, included as is in SGLang at `srt/eplb/eplb_algorithms/deepseek.py`) has two policies:

- **Hierarchical**: ① pack expert groups onto nodes balanced by load (each node gets the same number of groups); ② within each node, replicate the hottest experts several times (redundant experts); ③ pack the node's physical experts evenly onto its GPUs. Expert groups never cross nodes;
- **Global**: ignore nodes, replicate across all experts, and pack onto all GPUs.

Simulate two scales with this three-step algorithm (the same greedy rules as the open-source implementation):

```python
import numpy as np

rng = np.random.default_rng(0)
E, GROUPS, TOPK_GROUP, TOPK = 256, 8, 4, 8                 # DeepSeek-V3: 256 experts in 8 groups; each token first picks 4 groups, then 8 experts within them
GSIZE = E // GROUPS

bias = rng.normal(0, 1.0, E)                               # hot/cold differences between experts
scores = rng.random((20000, E)) + 0.15 * bias
grp = np.sort(scores.reshape(-1, GROUPS, GSIZE), -1)[..., -2:].sum(-1)      # group score: the sum of the top 2 scores in the group
keep = np.argsort(-grp, 1)[:, :TOPK_GROUP]
mask = np.full((len(scores), GROUPS), -np.inf)
np.put_along_axis(mask, keep, 0.0, 1)
routed = np.argsort(-(scores + np.repeat(mask, GSIZE, 1)), 1)[:, :TOPK]      # the 8 experts each token picks
load = np.bincount(routed.ravel(), minlength=E).astype(float)


def pack(weights, n_packs):
    """balanced_packing：从重到轻，每个物品放进"还有空位且当前最轻"的包，每包物品数相同"""
    per = len(weights) // n_packs
    packs, sums = [[] for _ in range(n_packs)], [0.0] * n_packs
    for i in np.argsort(-np.asarray(weights), kind="stable"):
        p = min((j for j in range(n_packs) if len(packs[j]) < per), key=sums.__getitem__)
        packs[p].append(i)
        sums[p] += weights[i]
    return packs


def replicate(ids, n_phys):
    """replicate_experts：反复给"单个副本负载最大"的专家再加一个副本"""
    cnt = {e: 1 for e in ids}
    for _ in range(n_phys - len(ids)):
        e = max(ids, key=lambda x: load[x] / cnt[x])
        cnt[e] += 1
    return [e for e in ids for _ in range(cnt[e])], cnt


def place(groups_to_nodes, GPUS, PHYS):
    """返回每张卡上的物理专家列表；groups_to_nodes=None 时不分层（整个集群当成一个节点）"""
    node_ids = ([sum((list(range(g * GSIZE, (g + 1) * GSIZE)) for g in gs), []) for gs in groups_to_nodes]
                if groups_to_nodes else [list(range(E))])
    gpus = []
    for ids in node_ids:
        phys, cnt = replicate(ids, PHYS * len(ids) // E)
        w = [load[e] / cnt[e] for e in phys]
        gpus += [[phys[i] for i in p] for p in pack(w, GPUS * len(ids) // E)]
    return gpus


def report(name, gpus, NODES):
    cnt = {}
    for g in gpus:
        for e in g:
            cnt[e] = cnt.get(e, 0) + 1
    gpu_load = [sum(load[e] / cnt[e] for e in g) for g in gpus]
    where = {}                                              # which nodes hold replicas of each expert
    for i, g in enumerate(gpus):
        for e in g:
            where.setdefault(e, set()).add(i // (len(gpus) // NODES))
    touched = []
    for row in routed[:2000]:                               # how many nodes each token goes to (replicas prefer nodes it already visits)
        nodes = set()
        for e in row:
            if not (where[e] & nodes):
                nodes.add(min(where[e]))
        touched.append(len(nodes))
    print(f"  {name}：最忙的卡是平均的 {max(gpu_load) / np.mean(gpu_load):.2f} 倍，每个 token 平均发往 {np.mean(touched):.2f} 个节点")


groups_load = load.reshape(GROUPS, GSIZE).sum(1)
for NODES, GPUS, PHYS in ((4, 32, 288), (8, 64, 320)):     # 32 / 64 redundant slots: 9 / 5 physical experts per GPU
    print(f"{NODES} 个节点、{GPUS} 张卡、{PHYS - E} 个冗余专家：")
    per = E // GPUS
    report("不做 EPLB（按编号放）", [list(range(i * per, (i + 1) * per)) for i in range(GPUS)], NODES)
    report("全局 EPLB（不看节点）", place(None, GPUS, PHYS), NODES)
    report("分层 EPLB（先把专家组打包到节点）", place(pack(groups_load, NODES), GPUS, PHYS), NODES)
```

```text title="output"
4 个节点、32 张卡、32 个冗余专家：
  不做 EPLB（按编号放）：最忙的卡是平均的 2.87 倍，每个 token 平均发往 3.18 个节点
  全局 EPLB（不看节点）：最忙的卡是平均的 1.03 倍，每个 token 平均发往 3.63 个节点
  分层 EPLB（先把专家组打包到节点）：最忙的卡是平均的 1.09 倍，每个 token 平均发往 3.27 个节点
8 个节点、64 张卡、64 个冗余专家：
  不做 EPLB（按编号放）：最忙的卡是平均的 3.97 倍，每个 token 平均发往 3.96 个节点
  全局 EPLB（不看节点）：最忙的卡是平均的 1.17 倍，每个 token 平均发往 5.21 个节点
  分层 EPLB（先把专家组打包到节点）：最忙的卡是平均的 1.79 倍，每个 token 平均发往 3.96 个节点
```

- **4 nodes** (2 groups per node): hierarchical EPLB brings the load down to 1.09× while preserving the "at most 4 nodes per token" cap; global EPLB balances better but scatters a group's experts across nodes, increasing cross-node traffic. This is exactly the scale of DeepSeek-V3's prefill (4 nodes, 32 GPUs, 8 experts + 1 redundant per GPU);
- **8 nodes** (only 1 group per node): the hierarchical policy can only balance at group granularity; whichever group is hot keeps its node busy, and the busiest GPU is 1.79× the average; the global policy balances much better, at the cost of 1.25 more nodes per token.

The open-source implementation's advice agrees: use the hierarchical policy when the number of nodes divides the number of groups and EP is small (prefill); use the global policy when EP is very large (decode).

In SGLang, `--ep-num-redundant-experts` sets the number of redundant experts, `--init-expert-location` specifies the initial placement (it can be an expert distribution measured offline), and `--enable-eplb` turns on rebalancing at runtime: `srt/eplb/expert_distribution.py` measures each expert's load, `eplb_manager.py` periodically calls the algorithm above, and `expert_location_updater.py` moves expert weights between GPUs.

## DP Attention load: the slowest rank sets the pace {#dp-attention-的负载最慢的-rank-决定速度}

Under DP Attention, each rank handles its own batch of requests. Attention time depends on the total KV of all requests on that rank, while the MoE layer's dispatch must wait for every rank to finish attention before exchanging tokens: **every layer has an implicit global synchronization**, and the slowest rank sets everyone's speed. Context lengths are long-tailed, so when one rank gets a few very long requests, the other ranks sit idle:

```python
import heapq

import numpy as np

rng = np.random.default_rng(1)
RANKS, PER_RANK, MAX_RUNNING = 32, 64, 128                # 32 DP ranks, 64 requests per rank on average, at most 128
ctx = np.minimum(rng.lognormal(np.log(3000), 1.0, RANKS * PER_RANK), 128000).astype(int)   # context length of each request


def attn_time(kv_tokens):                                 # one layer of attention (µs): read the attention weights + the latent KV of all this rank's requests
    return (187e6 + kv_tokens * 1152) / 3.35e12 * 1e6


def report(name, kv):
    t = [attn_time(x) for x in kv]
    print(f"{name}：注意力最慢的 rank {max(t):.0f} µs，平均 {np.mean(t):.0f} µs，其他 rank 平均空等 {max(t) - np.mean(t):.0f} µs")


report("轮流分配请求", [ctx[r::RANKS].sum() for r in range(RANKS)])
heap = [(0, r, 0) for r in range(RANKS)]                  # (total KV tokens, rank, requests)
for c in ctx:                                             # assign by total KV: give it to the rank with the least KV that is below the request limit
    skipped = []
    while True:
        kv, r, n = heapq.heappop(heap)
        if n < MAX_RUNNING:
            break
        skipped.append((kv, r, n))
    heapq.heappush(heap, (kv + int(c), r, n + 1))
    for s in skipped:
        heapq.heappush(heap, s)
report("按 KV 总量分配", [kv for kv, _, _ in heap])
```

```text title="output"
轮流分配请求：注意力最慢的 rank 243 µs，平均 165 µs，其他 rank 平均空等 78 µs
按 KV 总量分配：注意力最慢的 rank 183 µs，平均 165 µs，其他 rank 平均空等 18 µs
```

With round-robin assignment, nearly a third of each layer's attention time is spent waiting; assigning by total KV (ranks that got long requests take fewer requests) cuts the waiting to about a quarter of that. SGLang's `--load-balance-method` offers policies such as `round_robin`, `total_requests` and `total_tokens`. In real systems requests keep arriving and finishing, so assignment can only be approximately balanced, and there are supporting practices:

- **Idle ranks must also join every layer's all-to-all**: a rank with no requests at the moment must still run an "empty batch" to walk through every layer with the others, or the other ranks' dispatch waits on it forever;
- **CUDA Graph batches must line up**: ranks have different batch sizes, captured CUDA Graphs pad to buckets, and the padding is waste too;
- **Separate prefill and decode** (PD disaggregation): otherwise while one rank prefills a long prompt, every other rank's decode is held up.

Drag the spread of context lengths and compare the gap between the slowest rank and the average under the two assignment methods:

<div class="aig-widget" data-widget="dp-straggler"></div>

## Two-batch overlap {#双-batch-重叠}

Within one decode layer, the communication of dispatch and combine can take longer than the compute. **Two-batch overlap** (TBO) splits each rank's batch into two micro-batches: while one computes attention and experts, the other's tokens are on the network, alternating. But splitting is not free: decode's attention and expert GEMMs are both limited by **weight reads**, and after splitting in two, each micro-batch re-reads the weights. Estimate one decode layer of DeepSeek-V3 on H800:

```python
HBM, BF16, FP8, NIC = 3.35e12, 989e12, 1979e12, 50e9     # H800: memory bandwidth, BF16 / FP8 compute, NIC bandwidth per GPU
ATTN_W = 187e6                                          # weights of one MLA layer (FP8 bytes): q/kv down- and up-projections, output projection
EXPERT_W, EXPERT_FLOP = 44e6, 88e6                       # FP8 weight bytes of one expert; FLOPs for one token through one expert
LOCAL_EXPERTS, CTX = 3, 4096                             # 2 routed experts + 1 shared expert per GPU; average context length
TOK_IN, TOK_OUT = 7168 + 224, 7168 * 2                   # dispatch sends FP8 (with scales), combine returns BF16


def layer(b):
    """一张卡、一层、b 个 token 的各段时间（秒）：计算取访存与算力中较慢的一个"""
    attn = max((ATTN_W + b * CTX * 1152) / HBM, b * CTX * 278528 / BF16)         # read the attention weights + the latent KV
    moe = max(LOCAL_EXPERTS * EXPERT_W / HBM, b * 9 * EXPERT_FLOP / FP8)          # read the expert weights; b×8 routed + b shared
    return attn, moe, b * 8 * TOK_IN / NIC, b * 8 * TOK_OUT / NIC                 # the last two: dispatch, combine


print(" 每卡 batch   不重叠（µs/层）   双 batch 重叠   加速   计算 : 通信")
for b in (32, 64, 128, 256):
    attn, moe, disp, comb = layer(b)
    serial = attn + moe + disp + comb
    ha, hm, hd, hc = layer(b // 2)                      # split into two micro-batches: each re-reads the weights
    compute, comm = 2 * (ha + hm), 2 * (hd + hc)
    tbo = max(compute, comm)                            # ideally: while one micro-batch computes, the other communicates
    print(f"{b:>9}   {serial * 1e6:>14.0f}   {tbo * 1e6:>12.0f}   {serial / tbo:>4.2f}   {compute / comm:>5.2f}")
```

```text title="output"
 每卡 batch   不重叠（µs/层）   双 batch 重叠   加速   计算 : 通信
       32              252            236   1.07    2.12
       64              408            281   1.45    1.26
      128              732            445   1.65    0.83
      256             1409            890   1.58    0.65
```

- With a very small batch (32), compute is nearly all weight reads; splitting in two doubles the weight-read time, canceling the gain from overlap;
- With a batch between 64 and 128, where compute and communication take about as long, the gain is largest (close to the ideal 1.5–1.7×);
- With larger batches communication becomes the main part, overlap can only hide the compute, and the speedup starts to fall back; then the answer is less communication (FP8 dispatch, node-limited routing) or a faster network.

What the model leaves out: kernel launch and synchronization overhead (CUDA Graphs remove most of it), and the two micro-batches competing for SMs and memory bandwidth when running at the same time (which is exactly why DeepEP's low-latency mode uses a hook so communication takes no SMs). SGLang turns this on with `--enable-two-batch-overlap`; DeepSeek's published inference system uses a finer pipeline in the decode phase, splitting attention into two stages to form a five-stage pipeline that further balances the length of each stage. The prefill phase also uses two micro-batches: one's attention and expert compute overlapping with the other's dispatch / combine.

## Different EP sizes for prefill and decode {#prefill-和-decode-用不同规模的-ep}

The deployment in the DeepSeek-V3 paper:

| | prefill | decode |
| --- | --- | --- |
| Minimum deployment unit | 4 nodes, 32 GPUs | 40 nodes, 320 GPUs |
| Attention | TP4 + sequence parallelism, DP8 | TP4 + sequence parallelism, DP80 |
| MoE | EP32, 32 redundant experts (8 + 1 experts per GPU) | EP320, 1 expert per GPU, with 64 GPUs for redundant and shared experts |
| Communication | high-throughput all-to-all (NVLink forwarding within nodes) | direct point-to-point (IBGDA), low latency |

The reasons for such different sizes can all be found in earlier chapters:

- **Decode must feed the experts**: an expert GEMM's arithmetic intensity equals tokens per expert × 2 (FP8), so each expert needs hundreds of tokens per step ([FP8 and grouped GEMM](fp8-gemm.md)). Only DP attention over hundreds of GPUs pools a batch that large;
- **Decode must save memory for KV**: with only 1 expert per GPU, expert weights take only a small part of the tens of GB, and the rest of the memory all goes to the KV Cache, so each GPU runs more requests;
- **Prefill is compute-bound anyway**: one prefill already has thousands to tens of thousands of tokens, so a small EP is enough to feed the experts; small scale and few nodes also allow hierarchical EPLB to cap cross-node traffic;
- **PD disaggregation lets the two sides scale independently**: each side chooses its own instance count, EP size, parallelism and communication mode.

In an EP group of hundreds of GPUs, single-GPU failures and scaling become routine; how to keep one broken GPU from bringing down the whole group is covered in [fault tolerance, elastic scaling and troubleshooting for large-scale EP](ep-elastic.md).

!!! interview "In an interview"
    When a system design question asks you to "deploy a DeepSeek-scale MoE model", this chapter's content forms a complete answer: DP for attention and large-scale EP for MoE; with PD disaggregation, small EP for prefill (tens of GPUs, hierarchical EPLB, high-throughput all-to-all, two micro-batch overlap) and large EP for decode (hundreds of GPUs, global EPLB, low-latency all-to-all, few experts per GPU, memory left for KV); DP attention assigns requests by total KV with idle ranks running alongside; and two-batch overlap pays off most when compute and communication are comparable. Backing each point with a number is how to score high on this kind of question.

## Exercises {#练习}

**1. Memory for redundant experts.** DeepSeek-V3 has 58 MoE layers, and each expert's FP8 weights are about 44 MB. How much memory do the 32 redundant experts of the prefill deployment take in total? How much per GPU across 32 GPUs?

??? success "Answer"
    Each redundant "expert" has one copy in every MoE layer: $32 \times 58 \times 44\,\text{MB} \approx 81.7$ GB, about 2.55 GB per GPU across 32 GPUs. Not much against 80 GB of memory, and in exchange the busiest GPU drops from nearly 3× the average load to within 1.1×, nearly halving the latency of a layer.

**2. Why does decode use global EPLB?** Decode's EP is very large (hundreds of GPUs, tens of nodes), while the 256 experts form only 8 groups. What would the hierarchical policy do?

??? success "Answer"
    The hierarchical policy requires the number of nodes to divide the number of groups, and assigns groups to nodes as units. With tens of nodes and only 8 groups, either most nodes get no group, or it degrades into the global policy. Even with exactly 8 nodes, this chapter's simulation shows that group granularity only balances at the group level, leaving the busiest GPU at 1.79× the average. Decode places only 1–2 experts per GPU, so balancing must be as fine as single experts, hence the global policy, accepting more cross-node traffic (decode batches are small, so traffic is modest anyway, and the low-latency mode sends point-to-point directly regardless).

**3. The floor for two-batch overlap.** Using this chapter's model, write down the (approximate) condition under which two-batch overlap pays off.

??? success "Answer"
    Let the whole batch's compute time be $C(b)$ and communication time $M(b)$; split in two, compute becomes $2C(b/2)$ and communication about $M(b)$. The overlapped time is about $\max(2C(b/2), M(b))$, and it pays off when this is less than $C(b) + M(b)$. When compute is entirely limited by weight reads, $C(b/2) \approx C(b)$ and the condition becomes $\max(2C, M) < C + M$, i.e. $M > C$: communication must be longer than compute for overlap to make sense. With larger batches, where compute grows linearly with tokens, $2C(b/2) \approx C(b)$, the overlapped time approaches $\max(C, M)$, and the gain is largest.

## Summary {#小结}

- [x] Group-limited routing sends each token to at most 4 nodes; hierarchical EPLB (groups → nodes → replication within nodes → GPUs) preserves this cap and suits prefill with few nodes; global EPLB balances better and suits decode with very large EP.
- [x] DP Attention is synchronized by the all-to-all at every layer, so the slowest rank sets the pace; assigning requests by total KV, idle ranks running empty batches, and PD disaggregation all address this.
- [x] Two-batch overlap pays the price of re-reading weights, gains most when compute and communication are comparable (about 1.5×), and barely helps with batches that are too small.
- [x] Small EP for prefill, large EP for decode: decode must pool the global batch to feed the experts and leave memory for KV; PD disaggregation lets each side choose its size and communication mode independently.
