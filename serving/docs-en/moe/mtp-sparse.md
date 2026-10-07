# MTP and sparse attention: NSA and DSA

<p class="lead">This chapter covers two structures in the new generation of MoE models that are "designed for inference". <b>MTP</b> (multi-token prediction) has the model predict a few extra tokens during training, and serves as a speculative decoding draft at inference time; <b>sparse attention</b> (NSA, DSA) lets each query attend to only a small part of the past tokens. The general principles were covered in the <a href="../../topics/speculative/">advanced speculative decoding</a> and <a href="../../topics/long-context/">long context</a> chapters; here we focus on the accounting once they enter a large-scale MoE inference system: when MTP speeds things up and when it slows them down; what sparse attention saves and what it doesn't, and what new demands it places on inference engines.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What does DeepSeek-V3's MTP module look like? How is it used as a draft at inference time?
    2. In decode with large-scale EP, where does the cost of verifying several draft tokens fall?
    3. Why does the same MTP gain more with long contexts?
    4. What is DSA's "lightning indexer"? Sparse attention saves compute and bandwidth; does it save memory?
    5. Why is sparse attention actually unsuitable for prefill of short sequences?

??? success "Answers (try first, then expand to compare)"
    1. One layer chained after the main model: normalize the main model's last-layer hidden state and the embedding of "the next token" separately, concatenate them, project back to the hidden size, pass through a full Transformer layer, and predict the token after next with the output layer shared with the main model. At inference the main model produces one token per step, MTP guesses the next one, and the main model verifies them together in the next step.
    2. The per-request costs (attention weights, reading the latent KV) are amortized over the k+1 tokens and stay the same; the per-token costs (attention and expert compute, EP's all-to-all) multiply by k+1.
    3. With long contexts, the per-request KV reads are the bulk of each step, and that part is amortized over the verified tokens; the extra per-token part is relatively smaller, so the gain is larger.
    4. DSA's lightning indexer is a very light scorer (64 heads, FP8, ReLU-weighted) that scores all past tokens, and each query does MLA attention only over the 2048 highest-scoring tokens. It saves compute and bandwidth but not memory: every token's KV must still be kept, and the indexer's own keys take about 11% more.
    5. Short sequences have few tokens to begin with, so sparse selection saves little compute, while the indexer's scoring and top-k are extra overhead, so implementations fall back to dense attention.

## MTP: a draft grown on the model {#mtp长在模型身上的草稿}

DeepSeek-V3 chains an **MTP module** after the main model: it normalizes the main model's last-layer hidden state and the embedding of "the next token" separately, concatenates them, projects linearly back to the hidden size, passes through a full Transformer layer (MoE included), and finally predicts the token after next with the output layer **shared** with the main model. In training it provides an extra training signal; at inference time it is a ready-made draft model: the main model produces one token per step, the MTP module guesses the next one, and the next step the main model verifies both at once. The paper reports an acceptance rate of 85%–90% for the second token, raising generation speed (TPS) to 1.8×.

Usage in inference engines: SGLang uses `--speculative-algorithm NEXTN` (reusing the EAGLE implementation), and vLLM `--speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 1}'`. The MTP module has only one layer and can run autoregressively a few more times to produce several drafts, but it was only trained to predict "the one after next", so later drafts are accepted less often.

## In large-scale EP, where verifying tokens is expensive {#在大规模-ep-里验证-token-贵在哪}

The [advanced speculative decoding](../topics/speculative.md#什么时候有效) chapter used a dense 7B model to show that "the larger the batch, the smaller speculative decoding's gain". Decode with large-scale MoE EP has its own accounting: when verifying $k+1$ tokens,

- the **per-request** costs stay the same: attention weights and the latent KV are read only once, shared by the $k+1$ queries;
- the **per-token** costs multiply by $k+1$: attention compute, expert compute, and, most critically, **EP's all-to-all**, where every token must be sent to 8 experts and received back.

Reuse the single-layer model from the [previous chapter](ep-deploy.md#双-batch-重叠) (assuming two-batch overlap hides communication entirely behind compute, so each layer takes the longer of compute and communication), and compare MTP's gain across context lengths and batch sizes:

```python
HBM, BF16, FP8, NIC = 3.35e12, 989e12, 1979e12, 50e9
ATTN_W, EXPERT_W, EXPERT_FLOP, LOCAL_EXPERTS, LAYERS = 187e6, 44e6, 88e6, 3, 61
TOK_BYTES = (7168 + 224) + 7168 * 2                      # per token per expert: dispatch (FP8) + combine (BF16)
ACCEPT = [0.85, 0.75, 0.65]                              # acceptance rates of draft tokens 1, 2, 3 given all earlier ones were accepted (assumed)


def layer(seqs, tokens, ctx):
    """一张卡、一层：seqs 个请求、共 tokens 个 token（验证时每个请求 k+1 个），平均上下文 ctx"""
    attn = max((ATTN_W + seqs * ctx * 1152) / HBM, tokens * ctx * 278528 / BF16)   # the latent KV is read only once per request
    moe = max(LOCAL_EXPERTS * EXPERT_W / HBM, tokens * 9 * EXPERT_FLOP / FP8)
    return max(attn + moe, tokens * 8 * TOK_BYTES / NIC)  # assume two-batch overlap fully overlaps the all-to-all with compute


print("上下文   每卡请求数   不用 MTP（token/s/卡）   1 个草稿   2 个草稿   3 个草稿")
for ctx in (4096, 16384):
    for b in (16, 64):
        row = []
        for k in range(4):
            step = LAYERS * layer(b, b * (k + 1), ctx) + k * layer(b, b, ctx)   # verify k+1 tokens; the MTP module runs k times autoregressively
            adv, p = 1.0, 1.0
            for a in ACCEPT[:k]:
                p *= a
                adv += p                                 # expected tokens advanced: 1 + p1 + p1·p2 + ...
            row.append(b * adv / step)
        print(f"{ctx:>6}   {b:>10}   {row[0]:>20.0f}   " + "   ".join(f"{r / row[0]:>7.2f}x" for r in row[1:]))
```

```text title="输出"
上下文   每卡请求数   不用 MTP（token/s/卡）   1 个草稿   2 个草稿   3 个草稿
  4096           16                   2227      1.82x      1.72x      1.50x
  4096           64                   4716      0.92x      0.82x      0.72x
 16384           16                   1415      1.80x      1.73x      1.56x
 16384           64                   2302      1.30x      1.16x      1.01x
```

- **With few requests per GPU** (16), a step's time goes mostly to reading weights and KV, verifying one more token is nearly free, and 1 draft gets close to 1.8×, matching the figure reported in the paper;
- **With many requests per GPU and short contexts** (64, 4K), the all-to-all is already the bottleneck, verifying tokens doubles the communication, and MTP actually **slows things down**;
- **As the context grows** (16K), the "per-request" KV reads grow, and MTP again gains 1.3×.

So MTP is not "turn it on and it's faster": it pays off most for latency-sensitive scenarios with small per-GPU batches and long contexts; in throughput-oriented configurations where communication is already tight, use it with care, or with only 1 draft. Real systems also adjust the number of drafts dynamically with load, even turning speculative decoding off under high load. The acceptance rates, context lengths and network bandwidth in the model are assumptions; the direction of the conclusions matters more than the specific numbers.

## Sparse attention: NSA and DSA {#稀疏注意力nsa-与-dsa}

![Figure: two kinds of native sparse attention: NSA's three branches with gating, DSA's lightweight indexer with MLA](../assets/figures/nsa-branches.svg){.aig-svg}

The [long context](../topics/long-context.md#稀疏注意力每一步动态选择) chapter measured on a real model: if each query could pick exactly its 256 highest-scoring tokens, attention's error would be only about 9%. Attention itself is highly sparse; the hard part is "choosing quickly and accurately". Two schemes build sparsity in at training time:

- **NSA** (Native Sparse Attention): three attention branches merged with gates: the **compression** branch compresses each stretch of adjacent tokens into one coarse-grained key and value, giving a global overview; the **selection** branch uses the compression branch's attention scores to pick the most important **blocks** and does fine-grained attention within them; the **sliding window** branch covers the most recent context. Selection works in blocks, and query heads in the same GQA group share the selected blocks, so KV is read in large contiguous chunks, which suits the hardware;
- **DSA** (DeepSeek Sparse Attention, DeepSeek-V3.2): adds a **lightning indexer** on top of MLA. Each query token has 64 indexer heads of 128 dims each, and each past token has a 128-dimensional indexer key; the index score is $I_{t,s} = \sum_j w_{t,j}\, \text{ReLU}(q^I_{t,j} \cdot k^I_s)$, computed in FP8. Each query picks the top 2048 tokens by index score and does (absorbed-form) MLA attention only over them.

The indexer still has to score **every** past token, just at a much smaller cost per pair. Work out how much DSA saves relative to dense MLA:

```python
H, DC, ROPE = 128, 512, 64
PAIR_MLA = 2 * H * (DC + ROPE) + 2 * H * DC               # absorbed MLA: FLOPs per (query, key) pair
PAIR_EXPAND = 2 * H * (128 + ROPE) + 2 * H * 128          # expanded multi-head attention (used by dense prefill)
IDX_H, IDX_D, TOPK = 64, 128, 2048                        # DeepSeek-V3.2's lightning indexer: 64 heads, head dim 128; each query picks 2048 tokens
PAIR_IDX = 2 * IDX_H * IDX_D                              # indexer: FLOPs per (query, key) pair (FP8)
KV_BYTES, IDX_BYTES = (DC + ROPE) * 2, IDX_D + 4          # cached per token: the latent (bf16); the indexer key (FP8 + scale)
print(f"每对 (query, key)：MLA {PAIR_MLA / 1e3:.0f}K FLOPs，索引器 {PAIR_IDX / 1e3:.0f}K FLOPs；每个 token 的缓存多 {IDX_BYTES / KV_BYTES:.0%}")
print("上下文    decode 每层计算（稠密 → DSA）      decode 每层读取（稠密 → DSA）        prefill 每层计算（稠密 → DSA）")
for L in (4096, 32768, 131072):
    k = min(L, TOPK)
    dec = (L * PAIR_MLA, L * PAIR_IDX + k * PAIR_MLA)
    rd = (L * KV_BYTES, L * IDX_BYTES + k * KV_BYTES)
    pre = (L * L / 2 * PAIR_EXPAND, L * L / 2 * PAIR_IDX + L * k * PAIR_MLA)   # DSA's prefill also computes the selected tokens in absorbed form
    print(f"{L // 1024:>4}K   {dec[0] / 1e9:6.2f} → {dec[1] / 1e9:5.2f} GFLOPs（{dec[0] / dec[1]:4.1f} 倍）"
          f"   {rd[0] / 2**20:6.1f} → {rd[1] / 2**20:5.1f} MiB（{rd[0] / rd[1]:4.1f} 倍）"
          f"   {pre[0] / 1e12:7.1f} → {pre[1] / 1e12:6.1f} TFLOPs（{pre[0] / pre[1]:4.1f} 倍）")
```

```text title="输出"
每对 (query, key)：MLA 279K FLOPs，索引器 16K FLOPs；每个 token 的缓存多 11%
上下文    decode 每层计算（稠密 → DSA）      decode 每层读取（稠密 → DSA）        prefill 每层计算（稠密 → DSA）
   4K     1.14 →  0.64 GFLOPs（ 1.8 倍）      4.5 →   2.8 MiB（ 1.6 倍）       0.7 →    2.5 TFLOPs（ 0.3 倍）
  32K     9.13 →  1.11 GFLOPs（ 8.2 倍）     36.0 →   6.4 MiB（ 5.6 倍）      44.0 →   27.5 TFLOPs（ 1.6 倍）
 128K    36.51 →  2.72 GFLOPs（13.4 倍）    144.0 →  18.8 MiB（ 7.7 倍）     703.7 →  215.5 TFLOPs（ 3.3 倍）
```

- **It saves compute and bandwidth, not memory**: every past token's latent must be kept (the next query might pick any of them), plus an extra copy of the indexer's keys (11% more). The **capacity** problem of long contexts still has to be solved by KV offloading and tiered caching; sparse attention solves **speed**;
- **The longer the context, the better the deal**: at 128K, decode saves 13× in compute and nearly 8× in reads; at 4K it saves only about half;
- **Prefill of short sequences actually gets more expensive**: dense prefill uses expanded multi-head attention (82K FLOPs per pair), but under sparse attention each query picks different tokens, which only the absorbed MQA form can handle (279K FLOPs per pair); with a sequence of only 4K choosing 2048, the pairs saved do not make up for each pair being 3.4× more expensive. This is why the DeepSeek-V3.2 report implemented a dedicated "masked MHA mode" to emulate DSA for prefill of short sequences.

## New demands on inference engines {#对推理引擎的新要求}

- **Two KV caches**: besides the latent KV, the indexer's keys must also be stored in pages and take part in prefix caching and PD-disaggregation transfers, and the two must be handled together on eviction and offloading;
- **New kernels**: indexer scoring (small FP8 matrix multiplies + ReLU + weighted sums), per-query top-k selection (2048 out of tens of thousands, done efficiently on the GPU), and sparse MLA attention that reads only the selected tokens (FlashMLA and others provide sparse versions);
- **Reusing selections**: adjacent layers tend to select similar tokens, and some implementations let certain layers reuse the previous layer's top-k directly, skipping the indexer's compute (SGLang's DSA configuration has such a switch);
- **Scheduling and capacity planning**: sparse attention keeps long-context decode latency almost flat with length, but memory use grows all the same; the scheduler still admits requests by KV capacity, and capacity planning must treat "fast enough" and "fits" separately.

In the source: vLLM's `v1/attention/backends/mla/` has sparse MLA backends such as `flashmla_sparse.py` and `flashinfer_mla_sparse.py`, plus `indexer.py`; SGLang's `srt/layers/attention/` has `dsa_backend.py` and the `dsa/` directory.

!!! interview "In an interview"
    When asked about MTP, don't just say "speculative decoding speeds things up": explain that it is a one-layer draft trained along with the model with an 85%–90% acceptance rate, then give the conditions: of the cost of verifying tokens, the per-request part (weights, KV reads) is amortized, while the per-token part (compute, EP's all-to-all) multiplies by k+1, so small batches and long contexts gain the most, and it may slow things down when communication is tight. When asked about DSA, start with the indexer (64 heads, FP8, ReLU-weighted) + top-2048 selection, then the accounting: long-context decode saves an order of magnitude in compute but no memory at all (11% more, in fact), and prefill of short sequences is more expensive and falls back to the dense implementation.

## Exercises {#练习}

**1. How high must MTP's acceptance rate be to pay off?** In this chapter's model, with 64 requests per GPU and a 4K context, 1 draft makes a step take about 2× as long. How high must the acceptance rate be for MTP not to lose?

??? success "Answer"
    With 1 draft, each step advances $1 + p$ tokens in expectation and takes about $r$ times as long (here $r \approx 2$), so it breaks even when $(1 + p) / r \ge 1$, i.e. $p \ge r - 1 \approx 1$: the acceptance rate would have to reach 100% just to break even, which is impossible in practice. The reason is that in this configuration the all-to-all is already the bottleneck, and verifying tokens doubles the communication. Conclusion: in communication-bound decode configurations MTP can hardly bring any throughput gain, and is worth it only if single-request latency must come down.

**2. Sparse attention and prefix caching.** Two requests share a 100K prefix, served by a DSA model. Does prefix caching still work? What needs caching?

??? success "Answer"
    Yes. The prefix's latent KV and the indexer's keys depend only on the prefix itself and can be shared like ordinary prefix caching; both must be cached, hit together and evicted together. What cannot be cached is the top-k selection: it depends on each query, and a new request has different queries, selecting different tokens. Also, after the prefix hits, prefilling the new tokens still has the indexer score against the whole 100K prefix (16K FLOPs per pair), a cost that grows linearly with the prefix length.

## Summary {#小结}

- [x] MTP is a one-layer draft module trained along with the model, with an acceptance rate of about 85%–90% for the second token; enabled with NEXTN in SGLang and `deepseek_mtp` in vLLM.
- [x] In decode with large-scale EP, verifying tokens amortizes the per-request costs (weights, KV reads) but amplifies the per-token costs (compute, all-to-all); MTP gains most with small batches and long contexts, and may slow things down when communication-bound.
- [x] NSA combines three branches: compression, selection and a sliding window; DSA uses an FP8 lightning indexer to score all past tokens, and each query does MLA attention only over its top 2048.
- [x] Sparse attention saves compute and bandwidth but not memory (the indexer's keys add 11%); long-context decode gains the most, while prefill of short sequences falls back to the dense implementation.
- [x] Inference engines need: a paged cache for the indexer (taking part in prefix caching and transfers), kernels for scoring and top-k, sparse MLA kernels, and capacity planning that separates "fits" from "fast enough".
