# Case study: a published large-scale MoE inference system

<p class="lead">In early 2025, DeepSeek published the architecture of its V3 / R1 online inference system along with a day of operating statistics: how many machines it used, how many tokens each machine processed per second, the KV cache hit rate, the cost, and revenue at list prices. It is a rare source of real numbers on a large-scale MoE inference system. This chapter reviews it as a complete system design problem: first map each design to an earlier chapter, then check the published numbers with the earlier chapters' estimation methods: do the estimates hold, and where do they diverge?</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How large is the EP for prefill and for decode in this system? How many experts per GPU?
    2. From the published numbers, how many requests does each GPU serve concurrently in decode? Is there enough memory?
    3. How far is the decode step time estimated with the α-β and roofline models from the published data? Where does the gap come from?
    4. Roughly what compute utilization does prefill reach? Why isn't it higher?
    5. How was the published 545% "cost profit margin" calculated? What does it ignore?

??? success "Answers (try first, then expand to compare)"
    1. Prefill: 4 nodes, EP32, 9 routed experts + 1 shared expert per GPU; decode: 18 nodes, EP144, 2 routed experts + 1 shared expert per GPU (both with 32 redundant experts).
    2. About 88 concurrent requests per GPU, each with about 5000 tokens of KV on average, so about 31 GB of latent KV; add the attention weights (DP, one copy per GPU, about 11 GB) and the experts and vocabulary (about 10 GB), about 21 GB of weights, over 50 GB in all: it fits. MLA is the precondition for this deployment.
    3. The model gives a step lower bound of about 38 ms, with the all-to-all as the bottleneck; the published data works out to about 86 ms, 2.3× the bound. The gap is everything the model leaves out: NICs not reaching theoretical bandwidth, congestion and load imbalance in the all-to-all, kernels' actual efficiency, per-layer synchronization and scheduling overhead, imbalance between ranks, and not pushing the batch to its limit in order to guarantee 20 tokens per second.
    4. About 350 TFLOPS, 18% of FP8 peak. Reasons: the attention core and combine run in BF16, the overlap between all-to-all and compute is imperfect, uneven request lengths cause imbalance, and headroom must be left for time to first token rather than accumulating the largest batch.
    5. Take a day's cost at $2 per GPU-hour, convert every token (free traffic included) into revenue at R1's prices, and get a cost profit margin of about 5.5×. It ignores that most traffic is free, that V3 is priced lower, and that there are night-time discounts, so it is not actual revenue; the more useful conclusion is a hardware cost of about $0.5 per million output tokens.

## Architecture: which chapter each design maps to {#架构每个设计对应哪一章}

| Design | prefill | decode | Related chapter |
| --- | --- | --- | --- |
| Deployment unit | 4 nodes (32 GPUs) | 18 nodes (144 GPUs) | [large-scale EP deployment](ep-deploy.md) |
| Attention and shared experts | DP32 | DP144 | [MLA inference](mla.md), [expert parallelism and DP Attention](../distributed/expert-parallel.md) |
| Routed experts | EP32, 32 redundant experts, 9 routed experts + 1 shared expert per GPU | EP144, 32 redundant experts, 2 routed experts + 1 shared expert per GPU | [large-scale EP deployment](ep-deploy.md#eplb-与路由约束分层还是全局) |
| Communication | high-throughput all-to-all | low-latency all-to-all | [NVSHMEM and DeepEP](../comm/nvshmem-deepep.md) |
| Overlapping compute and communication | two micro-batches | a five-stage pipeline with attention split in two | [two-batch overlap](ep-deploy.md#双-batch-重叠) |
| Load balancing | balance attention compute and dispatched tokens across GPUs | balance KV usage and request counts across GPUs | [DP Attention load](ep-deploy.md#dp-attention-的负载最慢的-rank-决定速度) |
| Precision | FP8 matrix multiplies and dispatch, BF16 attention core and combine | same | [FP8 and grouped GEMM](fp8-gemm.md) |
| KV cache | on-disk KV cache (3FS) with a hit rate of about 56% | — | [KV transfer and storage](../comm/kv-storage.md) |
| Speculative decoding | — | MTP | [MTP and sparse attention](mtp-sparse.md) |

Two changes worth noting (compared with the earlier deployment in the paper): decode's EP shrank from 320 to 144, going from 1 expert per GPU to 2, a smaller deployment unit that scales with load more easily; and load balancing was split into three independent balancers (prefill, decode, experts), each solving a problem discussed in earlier chapters.

The published day of statistics (8 H800s per node): an average of 226.75 nodes occupied (278 at peak); 608 billion input tokens, 342 billion of them hitting the on-disk KV cache; 168 billion output tokens; an average output speed of 20–22 tokens/s, with each output token corresponding to 4989 KV entries on average; average throughput per node of 73.7 thousand input tokens/s for prefill (hits included), or 14.8 thousand output tokens/s for decode.

## Checking with the earlier chapters' methods {#用前几章的方法核对}

```python
GB = 1e9
# ---- published data (24-hour statistics, 8 H800s per node)
NODES_AVG, GPU_HOUR = 226.75, 2.0                     # average nodes occupied; cost at $2 per GPU-hour
IN_TOK, HIT_TOK, OUT_TOK = 608e9, 342e9, 168e9        # input tokens, those hitting the on-disk KV cache, output tokens
PREFILL_NODE, DECODE_NODE = 73.7e3, 14.8e3            # throughput per node: prefill input (hits included) / decode output, tokens/s
USER_TPS, AVG_CTX = 21, 4989                          # average output speed (tokens/s); average KV length per output token

# ---- 1. decode: how many requests per GPU, and does it fit in memory
per_gpu = DECODE_NODE / 8
seqs = per_gpu / USER_TPS
kv = seqs * AVG_CTX * 576 * 2 * 61                    # MLA latents, bf16, 61 layers
weights = 61 * 187e6 + 58 * 3 * 44e6 + 2 * 129280 * 7168    # attention (DP, one copy per GPU) + 2 routed experts and 1 shared expert + vocabulary
print(f"decode：每卡 {per_gpu:.0f} token/s，约 {seqs:.0f} 个并发请求；KV {kv / GB:.0f} GB + 权重 {weights / GB:.0f} GB")

# ---- 2. decode: estimate a lower bound on a step with the earlier chapters' single-layer model, versus the published speed
MTP_ADV = 1.8                                         # tokens advanced per step (MTP, 1 draft)
tokens, ctx = seqs * 2, AVG_CTX
attn = max((187e6 + seqs * ctx * 1152) / 3.35e12, tokens * ctx * 278528 / 989e12)
moe = max(3 * 44e6 / 3.35e12, tokens * 9 * 88e6 / 1979e12)
comm = tokens * 8 * ((7168 + 224) + 7168 * 2) / 50e9
step_model = 62 * max(attn + moe, comm)               # 61 layers + the MTP layer, assuming communication fully overlaps compute
step_real = MTP_ADV / USER_TPS
print(f"decode 一步：模型下限 {step_model * 1e3:.0f} ms（瓶颈：{'通信' if comm > attn + moe else '计算'}），"
      f"公开数据折算 {step_real * 1e3:.0f} ms，实际是下限的 {step_real / step_model:.1f} 倍")

# ---- 3. prefill: tokens actually computed and compute utilization
computed = PREFILL_NODE * (1 - HIT_TOK / IN_TOK) / 8
flop_per_tok = 2 * 37e9 + 61 * (AVG_CTX / 2) * 81920  # matrix multiplies over activated parameters + causal attention (expanded multi-head attention)
print(f"prefill：命中率 {HIT_TOK / IN_TOK:.1%}，每卡每秒真正计算 {computed:.0f} 个 token，"
      f"约 {computed * flop_per_tok / 1e12:.0f} TFLOPS（FP8 峰值的 {computed * flop_per_tok / 1979e12:.0%}）")

# ---- 4. cost and revenue at list prices
cost = NODES_AVG * 8 * 24 * GPU_HOUR
revenue = (HIT_TOK * 0.14 + (IN_TOK - HIT_TOK) * 0.55 + OUT_TOK * 2.19) / 1e6   # price per million tokens: cache hit / miss / output
print(f"每天成本 ${cost:,.0f}，按定价折算收入 ${revenue:,.0f}，成本利润率 {revenue / cost - 1:.0%}")
print(f"每百万输出 token 分摊的全部成本：${cost / (OUT_TOK / 1e6):.2f}")
```

```text title="output"
decode：每卡 1850 token/s，约 88 个并发请求；KV 31 GB + 权重 21 GB
decode 一步：模型下限 38 ms（瓶颈：通信），公开数据折算 86 ms，实际是下限的 2.3 倍
prefill：命中率 56.2%，每卡每秒真正计算 4030 个 token，约 348 TFLOPS（FP8 峰值的 18%）
每天成本 $87,072，按定价折算收入 $562,100，成本利润率 546%
每百万输出 token 分摊的全部成本：$0.52
```

Item by item:

**Memory.** About 88 concurrent requests per GPU with an average of 5000 KV entries is about 31 GB of latent KV; DP attention means every GPU holds a full copy of the attention weights (11 GB), which together with 3 experts and the vocabulary comes to about 21 GB of weights, over 50 GB in all: it fits, with room left for CUDA Graphs, activations and communication buffers. With a GQA model of the same size (hundreds of KB of KV per token), the same concurrency could not fit at all: MLA is the precondition for this deployment.

**Decode step time.** The model gives a lower bound of 38 ms, with the all-to-all as the bottleneck (176 tokens per GPU per step, each sent to 8 experts); the published data works out to about 86 ms per step, 2.3× the bound. The gap comes from what the model does not count: NICs not reaching the theoretical 50 GB/s, congestion and load imbalance of all-to-all at scale, kernels' actual efficiency, per-layer synchronization and scheduling overhead, imbalance across ranks (the slowest decides for everyone), and not pushing the batch to its limit in order to guarantee 20 tokens/s. The value of the estimate lies not in precision but in telling you **which item is the bottleneck and how far you are from the bound**: here the conclusion is "communication-bound, with roughly 2× in system overhead".

**Prefill utilization.** After removing the 56% that hit the cache, each GPU actually computes about 4000 tokens per second, about 350 TFLOPS, 18% of FP8 peak (equivalent to 35% of BF16 peak). Why not higher: the attention core and combine run in BF16, the overlap between all-to-all and compute is imperfect, uneven request lengths cause imbalance, and prefill instances must leave headroom for latency (time to first token) rather than accumulating the largest batch.

**Cost and revenue.** At $2 per GPU-hour, a day costs about $87 thousand; converting every token into revenue at R1's prices (free web and app traffic included) gives a "cost profit margin" of about 5.5×, consistent with the published 545% (the published token counts are rounded). The published material itself states this is not actual revenue: most traffic is free, V3 is priced lower, and there are night-time discounts. What the number really shows is: **at this level of optimization, the total hardware cost per million output tokens is about $0.5**, and cache hits (56%) cut the cost of input by more than half again.

## What is most worth learning from this system {#这套系统里最值得学的几条}

1. **First make the model architecture serve inference**: MLA squeezes KV down to 70 KB per token, which is what lets each GPU serve nearly a hundred requests under DP attention; MTP provides a free draft; fine-grained FP8 lets training and inference use the same precision.
2. **Split by phase and pick the best for each**: after PD disaggregation, prefill has small EP, high-throughput communication and two micro-batches; decode has large EP, low-latency communication and a five-stage pipeline.
3. **Serve MoE's need to "feed the experts"**: large-scale EP + DP attention pool the global batch; redundant experts and EPLB keep the busiest GPU from holding everyone back.
4. **Caching is a first-class citizen**: more than half the input tokens hit the on-disk KV cache, cutting the cost of input by more than half.
5. **Load balancing is everywhere**: attention compute, KV usage, dispatched tokens, expert load; any imbalance anywhere and the slowest GPU sets the speed for all.

!!! interview "How to explain it"
    To work through "design an online inference service for a DeepSeek-V3-scale model", you can use this chapter's framework directly: first state the deployment (PD disaggregation; prefill on 4 nodes with EP32, decode on 18 nodes with EP144; DP for attention, EP for experts, redundant experts), then estimate capacity (about 90 concurrent requests per GPU, 30 GB of KV, 20 GB of weights), then estimate latency (how much compute, reading and communication each decode step takes, with the all-to-all as the bottleneck), and finally cost (a hardware cost of about $0.5 per million output tokens, and the role of cache hits). Proactively state the assumptions and sources of error in your estimates: "my model gives a lower bound of 38 ms, and real systems are usually 1.5–2.5× the bound" is more credible than a seemingly precise number.

## Exercises {#练习}

**1. What if it were a GQA model?** An MoE model has 94 layers, 4 KV heads per layer and a head dimension of 128 (the shape of Qwen3-235B-A22B). How much KV per token? Under the same load of 88 requests per GPU with 5000 KV entries on average, how much memory does each GPU's KV need?

??? success "Answer"
    Per token: $94 \times 2 \times 4 \times 128 \times 2\,\text{B} \approx 188$ KB, 2.7× MLA's (70 KB). 88 requests × 5000 tokens × 188 KB ≈ 83 GB: an 80 GB GPU cannot even hold the KV, let alone the weights. So such models usually use TP for attention (KV heads split across GPUs; at TP=4 each GPU stores just 1 KV head) rather than DP attention, or lower the per-GPU concurrency and use FP8 KV. The model architecture directly decides the deployment.

**2. Find the next bottleneck.** If the NICs were upgraded to 800 Gb/s (double the bandwidth), what would become decode's bottleneck under this chapter's model? What would the step lower bound become?

??? success "Approach"
    Recompute with `comm` divided by 2: communication is about 306 µs per layer, while compute (attention about 248 µs + experts about 70 µs) is about 318 µs per layer, nearly even, and the bottleneck moves to attention compute (by then attention compute exceeds the time to read KV, so it is compute-bound). The step lower bound is about $62 \times 318\,\mu s \approx 20$ ms, down from 38 ms to about 20 ms. Going further requires work on attention: FP8 attention compute, sparse attention (DSA), which is exactly the direction later models took.

## Summary {#小结}

- [x] The published V3 / R1 inference system: PD disaggregation, prefill on 4 nodes with EP32, decode on 18 nodes with EP144, DP for attention and shared experts, EP for routed experts, 32 redundant experts, overlap via two micro-batches / a five-stage pipeline, three load balancers, and an on-disk KV cache hit rate of about 56%.
- [x] Checked with the earlier chapters' methods: decode serves about 88 concurrent requests per GPU with about 31 GB of KV, and MLA is the precondition for this deployment; the step lower bound is 38 ms with the bottleneck in the all-to-all, and the real step is about 2.3× the bound; prefill reaches about 18% of FP8 peak.
- [x] The 545% cost profit margin is a theoretical value at list prices; the more useful conclusion is a hardware cost of about $0.5 per million output tokens.
- [x] The value of estimates lies in locating the bottleneck and quantifying the gap, while proactively stating assumptions and sources of error.
