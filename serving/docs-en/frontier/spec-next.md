# New approaches to speculative decoding: block drafts, load-aware verification and PD disaggregation

<p class="lead">The <a href="../../topics/speculative/">speculative decoding</a> chapter covered the classic approach: a draft model proposes K tokens one at a time (EAGLE, MTP), the target model verifies them at once, and tree drafts improve the hit rate. It works well at low load, but slows things down once load rises, since verified tokens compete with real tokens for compute. Inference frameworks in 2026 improved it in three directions: <b>one forward pass producing a whole block of drafts</b> (PARD, DFlash, DSpark), <b>deciding how much to verify by load</b> (dynamic K, adaptive verification), and <b>cooperation with PD disaggregation and large-scale deployment</b>. This chapter ties these improvements together with a simple cost model.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why does the best draft length get shorter as the batch grows? When should you skip speculation altogether?
    2. How does DFlash produce a whole block of drafts in one forward pass? What does DSpark add on top, and why?
    3. How does vLLM's adaptive verification decide which draft tokens to verify? What extra does it need from the draft model?
    4. With PD disaggregation, what extra must the prefill instance send for the decode instance to start speculative decoding?
    5. Why must "adjusting K by load" be handled with special care together with data parallelism?

??? success "Answers (try first, then expand to compare)"
    1. Speculative decoding trades compute for steps: with small batches most of the GPU's compute is idle, so verifying a few more tokens is nearly free; with large batches compute is already tight, verified draft tokens compete with real tokens for compute, and the cost of extra verification exceeds the gain from acceptance. Once load is high enough (batch 512 for this chapter's 8B model), the best K is 0, and you should not speculate.
    2. DFlash projects the target model's hidden states into the draft model's context K and V, with queries only for "the next token + K masked positions", using non-causal attention within the block to produce the whole block in one forward pass. DSpark adds a Markov head on this backbone (adding a low-rank transition bias to the next position's logits based on the previously chosen token, restoring dependencies between positions in the block) and a confidence head (estimating each position's acceptance probability, for adaptive verification).
    3. Each (request, position) is scored by its survival probability, the running product of confidences along the positions; all positions of all requests are sorted by score and pass from highest to lowest, with how many pass decided by a cost model measured at startup to maximize "expected tokens produced / time of this step". It needs per-position confidence from the draft model (currently only DSpark has a confidence head).
    4. The target model's last-layer hidden state (the draft model's input) and the first round of drafts (such as the top-k), so the decode instance can start speculating as soon as it takes over, instead of running one step idle first.
    5. Under data parallelism (especially DP Attention + EP), ranks synchronize in every layer's all-to-all, and each rank's token count must match or be predictable; if each rank adjusts K by its own load, they verify different numbers of tokens and wait on each other. K must stay the same across all ranks.

## The higher the load, the smaller K should be {#负载越高k-越要小}

The gain of speculative decoding rests on one premise: decode is memory-bound, so computing a few more tokens in one forward pass costs almost nothing. With a large batch, one forward pass already holds many real tokens and becomes compute-bound, every extra draft token verified costs its share of compute, and the rejected ones are pure waste. Look at the best draft length $K$ with a simple cost model of an 8B model on an H100:

```python
# an 8B model on an H100: each step reads the weights once (the memory floor), and with many tokens it becomes compute-bound
W_BYTES, BW = 16e9, 3.35e12 * 0.8                  # bf16 weights, effective bandwidth
FLOP_PER_TOKEN, FLOPS = 2 * 8e9, 989e12 * 0.5      # FLOPs per token, effective compute
T_DRAFT = 0.25e-3                                  # one forward pass of the draft model (a one-layer EAGLE head or a small model, memory-bound)
ALPHA = 0.8                                        # probability the next position is accepted given all earlier drafts were


def step_time(tokens):
    return max(W_BYTES / BW, tokens * FLOP_PER_TOKEN / FLOPS)


def throughput(batch, k, parallel):
    """每秒生成的 token 数：每步 = 出草稿（逐个出 k 次 / 一次出整块）+ 验证（batch×(1+k) 个 token 一次前向）"""
    accepted = sum(ALPHA ** i for i in range(1, k + 1))          # expected accepted drafts; verification always yields 1 more on top
    draft = 0 if k == 0 else (T_DRAFT if parallel else k * T_DRAFT)
    return batch * (1 + accepted) / (draft + step_time(batch * (1 + k)))


for batch in (1, 32, 128, 512):
    base = throughput(batch, 0, False)
    best = {p: max(range(0, 9), key=lambda k: throughput(batch, k, p)) for p in (False, True)}
    print(f"batch {batch:>3}：不投机 {base:>7.0f} tok/s；逐个出草稿 最佳 K={best[False]}（{throughput(batch, best[False], False) / base:.2f} 倍）；"
          f"一次出整块 最佳 K={best[True]}（{throughput(batch, best[True], True) / base:.2f} 倍）")
```

```text title="output"
batch   1：不投机     168 tok/s；逐个出草稿 最佳 K=8（3.24 倍）；一次出整块 最佳 K=8（4.15 倍）
batch  32：不投机    5360 tok/s；逐个出草稿 最佳 K=5（2.95 倍）；一次出整块 最佳 K=5（3.41 倍）
batch 128：不投机   21440 tok/s；逐个出草稿 最佳 K=1（1.26 倍）；一次出整块 最佳 K=1（1.26 倍）
batch 512：不投机   30906 tok/s；逐个出草稿 最佳 K=0（1.00 倍）；一次出整块 最佳 K=0（1.00 倍）
```

- **The best K falls with load**: at batch 1, the larger K the better (the search stops at 8 here); at batch 128 only 1 remains; at batch 512 any draft is a loss. A deployment with a daytime peak and a night-time trough will have any fixed K wrong some of the time;
- **The cost of drafting one at a time grows linearly with K**: K drafts take K runs of the draft model, a sizable share of a step when K is large; a draft model that emits a whole block at once (next section) squeezes this into one forward pass, earning about 30% more at low load;
- At high load neither beats not speculating: this is the starting point for "adjusting by load" below.

Drag the batch and the acceptance rate to see how the best K moves:

<div class="aig-widget" data-widget="spec-load"></div>

## One forward pass for a whole block of drafts {#一次前向给出一整块草稿}

Drafting one at a time is slow because draft $i$ must wait for draft $i-1$. **Parallel drafting** lets the draft model predict the next K positions at once in one forward pass:

- **PARD** (Parallel Draft Model): put K "placeholder" tokens at the end of the draft model's input and train it to predict those K positions at once. In vLLM it is turned on with `"parallel_drafting": true` (for standalone draft models and EAGLE);
- **DFlash**: the draft model does not compute its own context; instead it projects the target model's hidden states into the context's K and V ("precomputed context KV"); the queries are just a small block of "the next token + K masked positions", with **non-causal** attention within the block (each position can see the other positions in the block), producing the whole block in one forward pass;
- **DSpark**: adds two small heads on DFlash's parallel backbone: a **Markov head** (a pair of low-rank matrices $V \times r$ and $r \times V$ that add a "transition bias" to the next position's logits based on the previously chosen token), sampled left to right in order to restore the dependencies between positions in the block; and a **confidence head** that estimates each position's acceptance probability, for the adaptive verification of the next section. DeepSeek-V4 released DSpark drafts (V4-Flash-DSpark), and vLLM's implementation (`models/deepseek_v4/nvidia/dspark.py`) borrows sparse attention's top-k indices, adding "future" query positions to each query's visible set to achieve non-causal attention within the block.

Why add a Markov head: positions predicted in parallel don't know what each other chose. If the next token has two equally likely continuations, position 2 has a different best answer on each path, and choosing independently can stitch together a draft "that doesn't fit together". Use a Markov chain as the target model to see the difference:

```python
import torch

torch.manual_seed(0)
V, K = 64, 6
# target model: a first-order Markov chain; each token is mostly followed by one of two "common continuations" (45% each), with the other 10% spread over the rest
P = torch.full((V, V), 0.1 / (V - 2))
for a in range(V):
    b, c = torch.randperm(V)[:2]
    P[a, b], P[a, c] = 0.45, 0.45


def expected_accept(draft, last):
    """目标模型按分布采样时，草稿被逐个接受的期望个数 = Σ_i（草稿前 i 个 token 恰好是目标生成的那条路的概率）"""
    p, total, prev = 1.0, 0.0, last
    for x in draft:
        p *= P[prev, x].item()
        total += p
        prev = x
    return total


def parallel_draft(last):
    """一次出整块、位置之间互不相干：第 i 个位置取 i 步之后的边缘分布里最可能的 token"""
    dist, out = torch.zeros(V), []
    dist[last] = 1
    for _ in range(K):
        dist = dist @ P
        out.append(int(dist.argmax()))
    return out


def markov_draft(last):
    """再用一个很小的"转移头"从左到右补上依赖：第 i 个位置看第 i-1 个位置选了什么"""
    out, prev = [], last
    for _ in range(K):
        prev = int(P[prev].argmax())
        out.append(prev)
    return out


par = sum(expected_accept(parallel_draft(t), t) for t in range(V)) / V
mar = sum(expected_accept(markov_draft(t), t) for t in range(V)) / V
print(f"每步期望接受的草稿数：各位置互不相干 {par:.2f}，加上位置间的转移 {mar:.2f}")
```

```text title="output"
每步期望接受的草稿数：各位置互不相干 0.62，加上位置间的转移 0.81
```

The Markov head is just a pair of low-rank matrices, so running it position by position costs almost nothing (vLLM replicates its weights on every rank to avoid an all-reduce per position); the bulk of the draft is still one forward pass. This is "semi-autoregressive": the heavy part in parallel, the light part in sequence.

## Deciding how much to verify by load {#按负载决定验证多少}

Three approaches at the framework level, from coarse to fine:

- **Dynamic K** (vLLM's `num_speculative_tokens_per_batch_size`): configure K piecewise by concurrency, for example 3 for 1–64 concurrent requests, 1 for 65–128, and no speculation beyond. RL rollouts suit it well: the batch starts large, and when only a few long-tail requests remain at the end, K grows automatically;
- **Adjusting by acceptance length** (SGLang's adaptive spec, `speculative/adaptive_spec_params.py`): tiers by batch size, each with a few candidate draft step counts, switching among candidates at runtime by the observed average acceptance length (with hysteresis to avoid flip-flopping); currently it supports EAGLE / EAGLE3 with topk=1;
- **Adaptive verification** (vLLM, currently only for DSpark with its confidence head): drafts still come as a whole block, but **each (request, position) decides separately whether to be verified**. A position's score is its **survival probability**, the probability that every earlier position is accepted, i.e. the running product of confidences; all positions of all requests are sorted together by score and pass from highest to lowest, with how many pass decided by a cost model measured at startup to maximize "expected tokens produced / time of this step".

Implement adaptive verification with the previous section's cost model, where half the requests have accurate drafts and half less so:

```python
import torch

torch.manual_seed(0)
K = 7                                               # the draft model gives 7 positions at once


def plan(conf):
    """每个 (请求, 位置) 的得分 = 存活概率（置信度沿位置累乘）；按得分从高到低放行 b 个，b 取让吞吐最大的那个"""
    batch = conf.shape[0]
    survival = conf.cumprod(1)                                  # each row is non-increasing, so what passes is always a prefix of each request
    order = survival.flatten().sort(descending=True).values
    gain = torch.cat([torch.zeros(1), order.cumsum(0)])         # expected extra accepted tokens when the top b positions pass
    rates = [(batch + gain[b]) / (T_DRAFT + step_time(batch + b)) for b in range(len(gain))]
    b = max(range(len(gain)), key=lambda i: rates[i])
    return b, rates[b], survival


for batch in (16, 128, 512):
    # half the requests have accurate drafts (code, restating), half less so (open-ended writing): per-position confidence
    conf = torch.cat([torch.full((batch // 2, K), 0.9), torch.full((batch - batch // 2, K), 0.55)]) * torch.rand(batch, K).mul(0.2).add(0.9)
    conf = conf.clamp(max=0.99)
    b, rate, survival = plan(conf)
    fixed = (batch + survival.sum()) / (T_DRAFT + step_time(batch * (1 + K)))
    none = batch / step_time(batch)
    kept = (survival >= survival.flatten().sort(descending=True).values[b - 1]).sum(1).float() if b else torch.zeros(batch)
    print(f"batch {batch:>3}：放行 {b:>4} 个草稿位置（准的请求平均 {kept[:batch // 2].mean():.1f} 个，不准的 {kept[batch // 2:].mean():.1f} 个）；"
          f"吞吐 不投机 {none:>6.0f}、全部验证 {fixed:>6.0f}、自适应 {rate:>6.0f} tok/s")
```

```text title="output"
batch  16：放行  112 个草稿位置（准的请求平均 7.0 个，不准的 7.0 个）；吞吐 不投机   2680、全部验证  10104、自适应  10104 tok/s
batch 128：放行   57 个草稿位置（准的请求平均 0.9 个，不准的 0.0 个）；吞吐 不投机  21440、全部验证  15182、自适应  29053 tok/s
batch 512：放行    7 个草稿位置（准的请求平均 0.0 个，不准的 0.0 个）；吞吐 不投机  30906、全部验证  15146、自适应  30448 tok/s
```

- **At low load, verify everything**: still in the memory-bound regime, so extra verification is nearly free;
- **At medium load, pick by request**: at batch 128 only 57 positions pass, almost all the first position of requests with accurate drafts; within the same step, one request verifies several tokens and another none at all, which a fixed K cannot do. Throughput is nearly double that of verifying everything, and 35% above not speculating;
- **At high load, verify almost nothing**: only the draft's own cost remains, and throughput is slightly below not speculating. So real deployments combine it with "don't draft at all at high load".

Adaptive verification's demands on the engine are concrete too (vLLM's documentation lists the restrictions): how many tokens each request verifies this step is decided on the GPU, and the CPU side knows only the upper bound, so **the attention backend must accept device-determined query lengths**; the step-time cost curve is measured at startup on the captured CUDA Graphs, so **full CUDA Graphs are required**, and `--enforce-eager` is not allowed; LoRA is not yet supported (LoRA's per-token mapping is built on the CPU), nor is pipeline parallelism (the cost curve and confidences exist only on the last stage).

## Cooperation with other mechanisms {#和其他机制的配合}

- **PD disaggregation**: once the decode instance has the KV and the first output token, it must start speculating immediately, which also needs the draft model's first-round input. SGLang has the prefill instance send the last position's hidden state and the draft's top-k along with the KV (the metadata buffer in `disaggregation/utils.py`), and the decode side assembles the draft input from them; EAGLE, DSpark and DFlash each have an implementation (`speculative/*_disaggregation.py`);
- **Draft layers without KV**: SGLang's Frozen-KV MTP lets the MTP draft layer read only the target model's KV Cache without keeping its own KV pool, saving the draft's share of memory;
- **Data parallelism**: when K changes, the number of tokens in a forward step changes. Data-parallel ranks schedule independently, and if each picks a different K, collectives across ranks (such as expert parallelism after DP attention) won't line up and may even deadlock. vLLM turns dynamic K off automatically when data parallelism is on; SGLang's adaptive spec supports neither DP attention nor two-batch overlap;
- **Stateful models**: compressor state (DeepSeek-V4) and linear-attention state (Kimi-K3, Qwen3-Next) are updated by draft tokens and must roll back on rejection; see the [new generation of open models](new-models.md) and [linear attention](linear-attn.md#投机解码与-pd-分离) chapters.

!!! interview "How to explain it"
    Start with the tension in principle: speculative decoding trades compute for steps; at low load compute is idle, at high load draft tokens compete with real tokens for compute, so the best K falls with batch size, and beyond some point you should not speculate; back it with numbers like "for an 8B model, K=8 is 3–4× faster at batch 1, and K=0 at batch 512". Then the three improvements: **block drafts** (PARD and DFlash emit K drafts in one forward pass; DSpark adds a low-rank Markov head to restore dependencies between positions, plus a confidence head); **load-aware verification** (dynamic K, SGLang adjusting draft steps by acceptance length, vLLM picking among all positions of all requests by survival probability, with the budget from a cost model measured at startup); **engineering cooperation** (PD disaggregation must transfer hidden states and draft top-k, K must be consistent under data parallelism, stateful models must be able to roll back).

## Exercises {#练习}

**1. Why is the set that passes, when sorted by survival probability, always a prefix of each request?**

??? success "Answer"
    The survival probability of a request's position $i$ is $c_1 c_2 \cdots c_i$, and each factor is at most 1, so it is non-increasing along the positions: position $i+1$ never scores higher than position $i$. When passing from highest score down, if a request's position $i+1$ passes, position $i$ has certainly already passed. This matches the semantics of verification exactly: once an earlier draft is rejected, verifying later ones is useless anyway.

**2. Why is the cost curve for adaptive verification measured at startup rather than computed by formula?**

??? success "Answer"
    Real step time is not just "the larger of memory and compute": attention time depends on context length and on sparse attention's indexer, CUDA Graphs are captured at a few fixed sizes (token counts are padded up to the nearest bucket, so time is stair-shaped), MoE time depends on the routing distribution, and it all differs across GPUs and parallel configurations. Measuring each size's time directly on the captured graphs at startup is most accurate. vLLM measures with an 8192-token context by default, and long-context deployments can raise it with `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` (with less effect for sparse-attention models like DeepSeek-V4, where what grows with context is mainly the cheap indexer).

**3. In the cost model above, change the draft model's forward pass from 0.25 ms to 1 ms. How does the best K change?**

??? success "Answer"
    Drafting one at a time now costs 1 ms per position, so the draft itself eats K's gains, and the best K at low load drops noticeably; a block draft pays only one extra 1 ms and is affected much less. This shows that the bigger the draft model, the clearer the advantage of parallel drafting: parallel drafts can afford a bigger, more accurate draft model and still pay off. Verify by changing `T_DRAFT` directly and running it.

## Summary {#小结}

- [x] Speculative decoding trades compute for steps: the best K falls with load, you should not speculate at high load, and any fixed K is wrong some of the time.
- [x] Block drafts: PARD and DFlash emit K drafts in one forward pass, and DSpark adds a low-rank Markov head to restore in-block dependencies plus a confidence head to estimate acceptance probabilities.
- [x] Load-aware verification: dynamic K, SGLang adjusting draft steps by acceptance length, and vLLM picking among all positions of all requests by survival probability, with the budget from a cost curve measured at startup.
- [x] Engineering cooperation: PD disaggregation must transfer hidden states and draft top-k, ranks must use the same K under data parallelism, and stateful models must be able to roll back draft updates.
