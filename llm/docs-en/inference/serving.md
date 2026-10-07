# Core concepts of inference serving

<p class="lead">Everything so far has been about "how one request is computed". A real inference service handles hundreds or thousands of requests at once, of different lengths, arriving and finishing at any time. This chapter ties together the core designs of inference engines such as vLLM and SGLang: metrics, continuous batching, KV cache management and prefix caching, chunked prefill, speculative decoding, PD disaggregation and parallelism. Every concept comes with a small runnable experiment.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What are TTFT, TPOT, throughput and goodput? How do they conflict?
    2. What is the difference between continuous and static batching? Why does the former have much higher throughput?
    3. Why does prefix caching save computation? How does it relate to RoPE and the KV cache?
    4. Why can speculative decoding speed things up without changing the output distribution? When does it work well?
    5. What is PD disaggregation? What problem does it solve?

??? success "Answers (try first, then expand to compare)"
    1. TTFT: time to first token (queueing + prefill); TPOT: the average time per token after that; throughput: tokens generated per unit time; goodput: throughput that meets the SLO. A larger batch raises throughput, but each step gets slower and queues get longer, so latency worsens: the two conflict.
    2. Static batching waits for the longest request in the batch to finish, so short requests hold their slots idle; continuous batching re-forms the batch at every step, finished requests leave immediately and new ones join immediately, keeping the GPU fully loaded with much higher throughput.
    3. The KV of an identical prefix has already been computed, so reusing it skips that part of the prefill. The KV stores K already rotated by position, so only a prefix at exactly the same positions can be reused, which is also why prefix caching can only match from the start.
    4. A cheap draft model guesses k tokens first, and the target model verifies them in parallel in one forward pass; with greedy verification the output is identical to generating one by one, and with sampling, rejection sampling keeps the distribution unchanged. It works well when the draft's acceptance rate is high (predictable output) and the batch is small (spare compute).
    5. Put prefill and decode on different instances (even different hardware) and transfer the KV between them: the two phases no longer interfere (a long prefill does not stall decode), and each can choose its own parallelism and machine ratio; the price is KV transfer and scheduling the ratio.

## Metrics {#指标}

| Metric | Meaning | Mainly affected by |
| --- | --- | --- |
| **TTFT** (Time To First Token) | time from a request's arrival to its first token | queueing time + prefill time |
| **TPOT** / **ITL** (Time Per Output Token / Inter-Token Latency) | the interval between subsequent tokens | each decode step's time, affected by batch size and context length |
| **End-to-end latency** | TTFT + TPOT × (output length − 1) | the sum of the two |
| **Throughput** | tokens generated (or requests handled) per unit time | higher with larger batches |
| **Goodput** | throughput of the requests that meet the latency requirements (the SLO, e.g. TTFT < 1 s and TPOT < 50 ms) | the balance of throughput and latency |

Throughput and latency inherently conflict: [the larger the batch](kv-cache.md#批处理让多个请求分摊权重读取), the higher the throughput, but the longer each request's TPOT; and when a new request's prefill is inserted, the requests in decode stall for a moment. Scheduling in an inference engine maximizes goodput under a given SLO.

## Continuous batching {#连续批处理}

**Static batching**: gather a batch of requests and run them until **all** of them finish, then switch to the next batch. The problem is that output lengths vary widely: a request that finishes after 20 tokens has to idle alongside one in the same batch that generates 500.

**Continuous batching (also called iteration-level scheduling)** (Orca, 2022): decide at **every step** which requests take part. Finished requests leave immediately and waiting ones fill in immediately, so the batch on the GPU stays as full as possible.

First, an animated comparison (static batching, continuous batching and chunked prefill; the same tool as in the inference systems book):

<div class="aig-widget" data-widget="contbatch"></div>

A simplified simulation shows the gap (a cost model for one decode step: a fixed 20 ms + 0.2 ms per request, matching "memory bound, so time barely grows with batch size"):

```python
import random

def simulate(policy, n_requests=300, max_batch=32, seed=0):
    rng = random.Random(seed)
    lengths = [rng.randint(10, 400) for _ in range(n_requests)]      # tokens each request will generate
    waiting = list(range(n_requests))
    running = {}                                                     # request id -> tokens remaining
    t, finish = 0.0, {}
    while waiting or running:
        if policy == "continuous" or not running:                    # static batching: refill only after the whole batch finishes
            while waiting and len(running) < max_batch:
                r = waiting.pop(0)
                running[r] = lengths[r]
        t += 20 + 0.2 * len(running)                                 # time of one decode step (ms)
        for r in list(running):
            running[r] -= 1
            if running[r] == 0:
                finish[r] = t
                del running[r]
    return sum(lengths) / (t / 1000), sum(finish.values()) / n_requests / 1000

for policy in ("static", "continuous"):
    tput, latency = simulate(policy)
    print(f"{policy:10s} 吞吐 {tput:6.0f} token/s，平均完成时间 {latency:6.1f} s")
static_tput, _ = simulate("static")
cont_tput, _ = simulate("continuous")
assert cont_tput > 1.5 * static_tput
```

```text title="output"
static     吞吐    696 token/s，平均完成时间   43.1 s
continuous 吞吐   1124 token/s，平均完成时间   27.4 s
```

Continuous batching has about 60% higher throughput, and the average completion time is more than a third shorter. The price is a complex implementation: each request in the batch has a different context length, keeps its KV cache in a different place and may be in a different phase (some in prefill, some in decode), which requires attention kernels that support "variable-length, paged" KV and a scheduler that manages memory step by step. This is exactly what vLLM's PagedAttention and scheduler solve.

## KV cache management and prefix caching {#kv-cache-管理与前缀缓存}

The [KV cache](kv-cache.md#kv-cache-的显存管理) chapter covered PagedAttention's block-table design. The most important optimization on top of it is **prefix caching**: many requests share the same beginning (a system prompt, the history of a multi-turn conversation, few-shot examples, the same document), and their KV caches are exactly the same (the same tokens at the same positions, with the same [RoPE rotations](../transformer/position.md)), so they need computing only once.

Check it on a real model: first compute the KV cache for a fairly long system prompt, then have two different questions reuse it directly:

```python
import copy
import time
import torch
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer, generate

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

system = "你是一个推理优化方面的专家助手。" + "回答时要简洁、准确，必要时给出数字依据。" * 10
questions = ["什么是 KV Cache？", "什么是连续批处理？"]
full_ids = [tok(tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids
            for q in questions]
# the common prefix of the two requests
n_prefix = 0
while full_ids[0][0, n_prefix] == full_ids[1][0, n_prefix]:
    n_prefix += 1

with torch.no_grad():
    prefix_cache = KVCache(model.cfg.num_hidden_layers)
    model(full_ids[0][:, :n_prefix], prefix_cache)                   # compute the prefix only once
    for ids in full_ids:
        cache = copy.copy(prefix_cache)                              # share the prefix KV (the tensors are never modified in place)
        cache.k, cache.v = list(prefix_cache.k), list(prefix_cache.v)
        reused = model(ids[:, n_prefix:], cache)[:, -1]              # compute only each request's own suffix
        full = model(ids)[:, -1]                                     # reference: full computation from scratch
        assert (reused - full).abs().max() < 1e-3
print(f"提示词共 {full_ids[0].shape[1]} 个 token，其中公共前缀 {n_prefix} 个；复用前缀后，每个请求只需 prefill "
      f"{full_ids[0].shape[1] - n_prefix} 个 token，结果与完整计算一致")
```

```text title="output"
提示词共 159 个 token，其中公共前缀 147 个；复用前缀后，每个请求只需 prefill 12 个 token，结果与完整计算一致
```

!!! inference "Inference view"
    vLLM hashes token content block by block to recognize reusable prefixes; SGLang's **RadixAttention** organizes all cached token sequences in a radix tree, which naturally supports many requests sharing common prefixes of any length, with LRU eviction. In multi-turn chat, agents (repeated calls with an ever-growing context) and batch evaluation (the same set of few-shot examples), the prefix cache hit rate can be very high, cutting TTFT sharply. The scheduler can also prioritize requests that hit the cache (cache-aware scheduling).

## Chunked prefill {#分块-prefill}

The prefill of a very long prompt (32K tokens, say) can take hundreds of milliseconds to seconds. Done in one go, every request in decode must wait during that time, and TPOT spikes. **Chunked prefill** cuts a long prefill into chunks (say 512 or 2048 tokens each), does one chunk per step, and computes it in the same batch as other requests' decode:

- Decode requests are no longer blocked by long prefills, so TPOT is steadier;
- Prefill chunks (compute intensive) and decode (memory intensive) are mixed, balancing hardware utilization.

Chunked prefill is mathematically identical to a one-shot prefill, because every chunk sees all previous chunks through the KV cache:

```python
ids = full_ids[0]
with torch.no_grad():
    one_shot = model(ids)[:, -1]
    cache = KVCache(model.cfg.num_hidden_layers)
    for start in range(0, ids.shape[1], 32):                         # prefill 32 tokens at a time
        chunk_logits = model(ids[:, start:start + 32], cache)
assert (chunk_logits[:, -1] - one_shot).abs().max() < 1e-3
```

## Speculative decoding {#投机解码}

Decode is slow because "each step computes only one token but reads all the weights". The idea of **speculative decoding** is to **guess** the next k tokens with a cheap method first, then have the large model verify all k positions **in one forward pass**. Verifying k tokens reads as many weights as generating 1 (the extra computation is nearly free when memory bound), so as long as enough guesses are right, one forward pass advances several tokens.

**Greedy verification**: the large model takes the argmax at each position, compares with the draft one by one, accepts the longest matching prefix, and adds the token the large model gives at the first mismatch. So **every verification advances at least 1 token, and the output is identical to ordinary greedy decoding**.

See how the draft's acceptance rate, width and depth decide the speedup with this tool (the same one as in the inference systems book):

<div class="aig-widget" data-widget="spectree"></div>

There are many ways to "guess": a small draft model (a small model of the same family), extra prediction heads built into the model (Medusa, EAGLE, DeepSeek-V3's [MTP](../training/pretraining.md#多-token-预测)), or the simplest, **n-gram lookup (prompt lookup)**: if the last few tokens appeared earlier in the context, take the tokens that followed them there as the draft. In tasks whose output repeats a lot of the input, such as restating, rewriting and code editing, its hit rate is very high:

```python
def truncate(cache, n):
    """丢掉被拒绝的草稿 token 的 KV。"""
    for layer in range(len(cache.k)):
        cache.k[layer] = cache.k[layer][:, :, :n]
        cache.v[layer] = cache.v[layer][:, :, :n]

def ngram_draft(seq, k, n=3):
    """在上文中查找最后 n 个 token 上一次出现的位置，把紧随其后的 k 个 token 作为草稿。"""
    tail = seq[-n:]
    for start in range(len(seq) - n - 1, -1, -1):
        if seq[start:start + n] == tail:
            return seq[start + n:start + n + k]
    return []

@torch.no_grad()
def speculative_generate(model, ids, max_new, k=6, eos=None):
    cache = KVCache(model.cfg.num_hidden_layers)
    nxt = model(ids, cache)[0, -1].argmax().item()
    seq, out, target_calls = ids[0].tolist(), [], 1
    while len(out) < max_new:
        out.append(nxt)
        seq.append(nxt)
        if nxt == eos:
            break
        draft = ngram_draft(seq, k)
        base = cache.length
        preds = model(torch.tensor([[nxt] + draft]), cache)[0].argmax(-1).tolist()   # verify all draft tokens in one forward pass
        target_calls += 1
        n_ok = 0
        while n_ok < len(draft) and preds[n_ok] == draft[n_ok] and len(out) < max_new:
            out.append(draft[n_ok])
            seq.append(draft[n_ok])
            n_ok += 1
            if out[-1] == eos:
                break
        if out[-1] == eos:
            break
        truncate(cache, base + 1 + n_ok)                              # roll back the rejected draft tokens
        nxt = preds[n_ok]                                             # the large model's prediction at the first mismatch
    return out, target_calls

para = "推理服务的调度器在每一步决定哪些请求参与计算。连续批处理允许新请求在任意一步加入，已完成的请求立即离开，从而保持较高的硬件利用率。"
msgs = [{"role": "user", "content": "请把下面这段话原样重复一遍，不要做任何修改：\n" + para}]
ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids

t0 = time.perf_counter()
greedy = generate(model, ids, 80, eos_token_id=tok.eos_token_id)[0].tolist()
t_greedy = time.perf_counter() - t0
t0 = time.perf_counter()
spec, calls = speculative_generate(model, ids, 80, eos=tok.eos_token_id)
t_spec = time.perf_counter() - t0
assert spec == greedy                                                 # output identical to plain greedy, token by token
print(f"生成 {len(greedy)} 个 token：普通解码 {len(greedy)} 次前向（{t_greedy:.2f} s），"
      f"投机解码 {calls} 次前向（{t_spec:.2f} s）")
```

In this handbook's environment:

```text
生成 38 个 token：普通解码 38 次前向（1.83 s），投机解码 8 次前向（0.58 s）
```

38 tokens took only 8 forward passes of the large model, with identical output.

### Speculative decoding under random sampling {#随机采样时的投机解码}

With sampling, comparing argmaxes is not enough. Standard **speculative sampling** (Leviathan et al., Chen et al., 2023) uses rejection sampling to keep the output distribution exactly that of the large model: with the draft model's distribution q and the large model's p, for a draft token x:

- accept it with probability $\min(1, p(x)/q(x))$;
- if rejected, resample a token from the corrected distribution $\text{norm}(\max(0, p - q))$, and the round ends.

Verify "the output distribution equals p" with a Monte Carlo simulation:

```python
torch.manual_seed(0)
p = torch.tensor([0.5, 0.3, 0.15, 0.05])          # the large model's distribution
q = torch.tensor([0.25, 0.25, 0.25, 0.25])        # the draft model's distribution (a poor draft)

def speculative_sample(p, q):
    x = torch.multinomial(q, 1).item()                               # the draft model proposes
    if torch.rand(()) < min(1.0, (p[x] / q[x]).item()):
        return x, True                                               # accept
    residual = (p - q).clamp(min=0)
    return torch.multinomial(residual / residual.sum(), 1).item(), False

n = 200_000
samples, accepted = zip(*(speculative_sample(p, q) for _ in range(n)))
freq = torch.bincount(torch.tensor(samples), minlength=4).float() / n
print("输出分布", [round(v, 3) for v in freq.tolist()], " 目标分布", [round(v, 3) for v in p.tolist()],
      f" 接受率 {sum(accepted) / n:.3f}（理论值 Σmin(p,q) = {torch.minimum(p, q).sum().item():.3f}）")
assert (freq - p).abs().max() < 0.01
```

```text title="output"
输出分布 [0.499, 0.3, 0.151, 0.05]  目标分布 [0.5, 0.3, 0.15, 0.05]  接受率 0.699（理论值 Σmin(p,q) = 0.700）
```

Even with a very poor draft, the output distribution matches the large model exactly; only the acceptance rate is low and the speedup small. The **acceptance rate** $\sum_x \min(p(x), q(x))$ measures how close the draft is to the large model and directly decides the speedup.

!!! inference "Inference view"
    Speculative decoding pays off most with **small batches and low latency**: decode is then severely memory bound, and verifying several tokens adds almost no time. With very large batches the GPU is already near the compute limit, the extra draft tokens are no longer "free", and the gain shrinks or even turns negative. It also requires rolling back the KV cache (the KV of rejected draft tokens must be discarded) and working with CUDA Graphs and paged KV, which is quite complex to engineer. The EAGLE family and MTP are currently the most common schemes in inference engines.

## PD disaggregation {#pd-分离}

Prefill is compute intensive and decode memory intensive, and putting them on the same GPUs makes them interfere: new requests' prefills make decode stutter (TPOT jitter), and decode's many small steps slow prefill down (longer TTFT). **PD disaggregation (prefill-decode disaggregation)** puts the two phases on different GPUs (even different machines):

1. A prefill instance processes the prompt and produces the KV cache;
2. The KV cache is transferred over a fast network (NVLink, RDMA) to a decode instance;
3. The decode instance continues generating.

The two kinds of instances can use different parallelism, batch sizes and even different GPU models, optimizing TTFT and TPOT separately. The price is the KV cache transfer and system complexity. Work such as DistServe, Splitwise and Mooncake (Kimi) pushed this direction forward, and vLLM, SGLang and NVIDIA Dynamo all support it.

## Parallelism in inference {#推理中的并行}

| Parallelism | What it splits | Communication | Suited for |
| --- | --- | --- | --- |
| Tensor parallelism (TP) | each layer's matrices across GPUs, by head / by dimension | 2 all-reduces per layer | within a node (NVLink): lower single-request latency, fit large models |
| Pipeline parallelism (PP) | by layer | point-to-point between adjacent stages | across nodes: little communication, but no lower single-request latency |
| Data parallelism (DP) | a full replica per GPU (or per group of GPUs) | none | scaling throughput |
| Expert parallelism (EP) | MoE experts across GPUs | 2 all-to-alls per MoE layer | MoE models |
| DP attention | data parallelism by request for attention, expert parallelism for MoE | all-to-all / all-gather | models like MLA whose KV cannot be split by head (DeepSeek) |

## Putting it together: what makes up an inference engine {#把它们串起来一个推理引擎的组成}

<!-- i18n:diagram 7a50b690e3 -->
```text
API server (OpenAI-compatible interface)
  │ tokenize, apply the chat template
  ▼
Scheduler ────────────────────────────── decides at every step: which requests decode, which new requests prefill (in chunks),
  │                                      whether to preempt; allocates KV cache blocks for them, looks up the prefix cache
  ▼
Model executor (one worker per GPU)
  │ builds batch inputs (variable-length sequences, block tables, positions)
  │ forward: fused operators + attention backend (FlashAttention / FlashInfer) + quantized GEMMs + TP/EP communication
  │ decode uses CUDA Graphs
  ▼
Sampler (temperature, top-p, penalties, constrained decoding, speculative verification)
  ▼
Detokenize and stream back
```

The main source directories of vLLM and SGLang map almost one to one onto this picture. SGLang also overlaps the CPU work of "scheduling the next batch" with the GPU work of "running the current batch" (the overlap scheduler), further reducing GPU idle time.

!!! interview "In an interview"
    Asked about the core metrics and techniques of inference serving: first define TTFT (queueing + prefill), TPOT (per decode token) and goodput (throughput that meets the SLO), explain that throughput and latency conflict, and that scheduling aims to maximize goodput under the SLO; then list the techniques and say which metric each addresses: continuous batching (step-by-step scheduling, raises throughput), prefix caching and chunked prefill (mathematically equivalent; lower TTFT, steadier TPOT), speculative decoding (greedy verification does not change the output, rejection sampling does not change the distribution; lowers TPOT), PD disaggregation (the two phases on different hardware), and the parallelisms TP, PP, DP, EP and DP attention.

## Exercises {#练习}

**1. Reading an SLO.** A service's SLO is TTFT < 500 ms and TPOT < 40 ms. At peak hours you find TPOT often exceeds the target while TTFT is fine. What are the likely causes and remedies?

??? success "Approach"
    TPOT over target usually means each decode step is too slow: the batch is too large (too much KV to read per step), contexts are too long, or large prefills are cutting in and causing stalls. Remedies: cap the batch size (trading throughput for latency); enable chunked prefill or lower its chunk size so long prefills do not block decode; quantize the KV cache to read less; enable speculative decoding to cut per-token latency; if prefill and decode interfere badly, consider PD disaggregation. Then confirm the effect of each change with monitoring.

**2. Estimating acceptance and speedup.** Speculative decoding proposes k = 4 draft tokens per round, each accepted with probability α = 0.7 (independently; once one is rejected, all later ones are discarded). How many tokens does each round advance on average?

??? success "Answer"
    The tokens advanced per round = accepted draft tokens + 1 (the large model's token at the first rejection, or the next token after all are accepted). The expected value is $\sum_{i=0}^{k} \alpha^i = (1 - \alpha^{k+1}) / (1 - \alpha)$:

    ```python
    alpha, k = 0.7, 4
    print(f"{(1 - alpha ** (k + 1)) / (1 - alpha):.2f}")   # about 2.77
    ```

    Each forward pass of the large model advances about 2.77 tokens on average. The real speedup must still subtract the cost of generating the draft and the extra computation of verifying several tokens.

## Summary {#小结}

- [x] TTFT reflects prefill and queueing, TPOT reflects decode; throughput and latency conflict, and scheduling aims to maximize goodput under the SLO.
- [x] Continuous batching schedules step by step, with finished requests leaving and new ones joining immediately, for throughput far above static batching.
- [x] Prefix caching reuses the KV cache of identical prefixes; chunked prefill keeps long prefills from blocking decode; both are mathematically equivalent to the original computation.
- [x] Speculative decoding advances several tokens with a cheap draft plus one verification; greedy verification leaves the output unchanged, and rejection sampling keeps the sampling distribution unchanged.
- [x] PD disaggregation puts the two phases of different character on different hardware; inference parallelism includes TP, PP, DP, EP and DP attention.

For the related math (the proof that the speculative decoding acceptance rate = 1 − total variation distance, see [probability and sampling](math://probability/); Little's law, queueing theory and confidence intervals for P99, see [math in performance and serving](math://performance-math/)).
