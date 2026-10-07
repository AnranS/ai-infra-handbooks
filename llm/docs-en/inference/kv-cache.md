# The KV cache and two-phase inference

<p class="lead">The KV cache is the single most important concept in large-model inference: it brings the computation of generation down from quadratic to linear, and it brings with it nearly every memory management problem in inference systems. Understand it, and you understand why inference splits into two phases of entirely different character, prefill and decode, and why batching raises throughput so much.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why can the K and V of past tokens be cached and reused? Why does Q not need caching?
    2. Generating n tokens without a KV cache, how many tokens are processed in total? And with one?
    3. What are the arithmetic intensities of prefill and decode? What is each one's bottleneck?
    4. Why does decoding 8 requests together take far less time than decoding each of them separately?
    5. Which KV cache problem does PagedAttention solve?

??? success "Answers (try first, then expand to compare)"
    1. In causal attention, a past token's K and V depend only on itself and earlier tokens, so adding new tokens later does not change them; Q is used only once, to compute the current token's output, and is never needed again.
    2. Without a cache, generating token t reprocesses t tokens, about $n^2/2$ in total for n tokens; with a cache each step processes only 1 new token, about n in total (plus one prefill of the prompt).
    3. Prefill processes hundreds or thousands of tokens at once, so its arithmetic intensity is about the number of tokens: compute bound. Decode processes 1 token per request per step, so its intensity is about the batch size, far below the ridge point: memory bound.
    4. Decode time goes mostly to reading the weights; when 8 requests decode together, the weights are read once while 8 tokens are computed, so the time grows very little (only 60% more in this chapter's measurement) and throughput becomes about 5 times as high.
    5. Fragmentation and waste in KV cache memory: reserving contiguous space for the maximum length leaves lots of internal and external fragmentation; with paging, blocks are allocated on demand, and prefixes can even be shared between requests block by block.

<!-- comic ../assets/comics/kv-cache.webp is in Chinese; put it back once the English version exists -->

## Why K and V can be cached {#为什么-kv-可以缓存}

To generate token t+1, does the model have to run the full forward pass over the first t tokens? Look at the structure of causal attention: **the output at position i depends only on the inputs at positions ≤ i**. So when a new token is appended, the hidden states of all past positions in every layer do not change, and neither do their K and V. The new token only needs to:

1. Compute its own Q, K and V;
2. Append its K and V to the cache;
3. Attend with its own Q to **all** the K and V in the cache.

Past tokens' Q is never used again (only the newest token's query takes part in the computation), so only K and V are cached. This is the **KV cache**.

Without a cache, generating token t reprocesses t tokens, about $n^2/2$ tokens in total for n tokens; with a cache, each step processes only 1 new token.

## Check: step-by-step computation with a cache equals full recomputation {#验证带缓存的逐步计算等于完整重算}

This is the most common way to check that an inference implementation is correct:

```python
import time
import torch
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer

path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
ids = tok("KV Cache 是大模型推理中最重要的概念之一，因为", return_tensors="pt").input_ids

with torch.no_grad():
    full = model(ids)                                   # process the whole sequence at once
    cache = KVCache(model.cfg.num_hidden_layers)
    step_logits = [model(ids[:, :5], cache)]            # prefill the first 5 tokens first
    for t in range(5, ids.shape[1]):                    # then decode the remaining tokens one by one
        step_logits.append(model(ids[:, t:t + 1], cache))
    incremental = torch.cat(step_logits, dim=1)

diff = (full - incremental).abs().max().item()
print(f"逐步计算 vs 完整重算，logits 最大差异 {diff:.1e}；缓存长度 {cache.length}，K 的形状 {tuple(cache.k[0].shape)}")
assert diff < 1e-3 and cache.length == ids.shape[1]
```

Each layer's K in the cache has shape `[B, n_kv, S, d_h]` = `[1, 8, S, 128]`.

## Prefill and decode {#prefill-与-decode}

With a KV cache, a generation request splits into two phases:

![Figure: prefill and decode](../assets/figures/kv-cache.svg){.aig-svg}

| | Prefill | Decode |
| --- | --- | --- |
| Input | the whole prompt (T tokens) | 1 new token per step |
| What it does | computes the K and V of all prompt tokens, writes them to the cache, and outputs the first new token | reads the cache, generates the next token, and appends its K and V |
| Tokens computed per weight read | T | the number of requests in the batch |
| Bottleneck | compute (when T is large) | memory bandwidth (reading weights and the KV cache) |
| Latency metric | TTFT (time to first token) | TPOT / ITL (latency per output token) |

The difference in **arithmetic intensity** is the key to everything. In a BF16 linear layer, each parameter is 2 bytes, and each token uses it for 2 operations:

- decode (batch = 1): 2 operations per 2 bytes read, an arithmetic intensity of about 1 FLOP/byte;
- prefill (T tokens): 2T operations per 2 bytes read, an arithmetic intensity of about T FLOP/byte.

The H100's ridge point is about 989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/byte. So a prefill of a few hundred tokens saturates the GPU's compute, while decode falls two orders of magnitude short: **most of the GPU's compute sits idle during decode**, and the time goes to reading the weights and the KV cache from memory.

The same pattern shows up on a CPU (a CPU has the same balance problem of "memory bandwidth vs. compute"):

```python
torch.set_num_threads(16)
prompt = tok("介绍一下大模型推理中的 prefill 和 decode 两个阶段。" * 8, return_tensors="pt").input_ids

@torch.no_grad()
def timed_generate(input_ids, n_new=20):
    cache = KVCache(model.cfg.num_hidden_layers)
    t0 = time.perf_counter()
    logits = model(input_ids, cache)
    t_prefill = time.perf_counter() - t0
    nxt = logits[:, -1].argmax(-1)
    t0 = time.perf_counter()
    for _ in range(n_new):
        nxt = model(nxt[:, None], cache)[:, -1].argmax(-1)
    return t_prefill, (time.perf_counter() - t0) / n_new

timed_generate(prompt[:, :8], n_new=2)                   # warm-up
t_pre, t_dec = timed_generate(prompt)
T = prompt.shape[1]
print(f"prefill {T} 个 token: {t_pre * 1000:.0f} ms（每个 token {t_pre / T * 1000:.1f} ms）")
print(f"decode: 每个 token {t_dec * 1000:.1f} ms")
```

Running it in this handbook's environment gives (the numbers vary by machine; the ratio is what matters):

```text
prefill 104 个 token: 179 ms（每个 token 1.7 ms）
decode: 每个 token 45.5 ms
```

Prefill averages only 1.7 ms per token, while decode takes 45.5 ms to generate one token, about 26 times more: for the same single read of the weights, prefill spreads it over more than a hundred tokens and decode over just one.

## Batching: letting requests share the weight reads {#批处理让多个请求分摊权重读取}

Since decode's bottleneck is "one weight read computes only one token", let several requests decode **together**: read the weights once and compute one token for each of B requests. As long as compute is not yet the bottleneck, the time barely grows with B while throughput rises B-fold:

```python
@torch.no_grad()
def decode_step_time(batch, ctx_len=64, steps=10):
    ids = torch.randint(0, 150000, (batch, ctx_len))
    cache = KVCache(model.cfg.num_hidden_layers)
    model(ids, cache)
    nxt = torch.randint(0, 150000, (batch,))
    t0 = time.perf_counter()
    for _ in range(steps):
        nxt = model(nxt[:, None], cache)[:, -1].argmax(-1)
    return (time.perf_counter() - t0) / steps

decode_step_time(2, steps=2)                              # warm-up
t1, t8 = decode_step_time(1), decode_step_time(8)
print(f"batch=1: 每步 {t1 * 1000:.1f} ms；batch=8: 每步 {t8 * 1000:.1f} ms（{t8 / t1:.1f} 倍的时间，8 倍的 token）")
```

In this handbook's environment:

```text
batch=1: 每步 45.4 ms；batch=8: 每步 74.8 ms（1.6 倍的时间，8 倍的 token）
```

With 8 requests decoding together, each step takes only 60% longer, while throughput rises to about 5 times. **This is the fundamental reason inference services pursue large batches**. But the batch cannot grow without limit: every request has its own KV cache, so memory runs out first; and the larger the batch, the higher each request's decode latency. Scheduling in an inference service is essentially a balance between throughput and latency; see [serving](serving.md).

!!! inference "Inference view"
    In decode, besides the weights, **the KV cache must also be read in full at every step**. The weights are shared by all requests, so batching amortizes them; the KV cache is private to each request and cannot be amortized. The longer the context and the larger the batch, the larger the share of time spent reading the KV cache, until it can exceed the time spent reading the weights. That is why long-context inference is slow, and why GQA, MLA and KV cache quantization matter so much (see [attention variants](../transformer/attention-variants.md)).

Work this out for concrete models and GPUs: pick a model, a GPU and an average context length, and see how many requests the KV fits and how long a decode step takes at least at full load.

<div class="aig-widget" data-widget="kv-calc"></div>

A few combinations worth trying: LLaMA-3-8B and Qwen2.5-7B have similar parameter counts, but their KV head counts differ by a factor of two and the former has 4 more layers, so their KV per token differs by more than double, and so does the number of requests they can serve; switching the KV to FP8 doubles the number of requests; DeepSeek-V3's FP8 weights do not fit on 8 H100s but do on 8 H200s, and every GPU has to store its own copy of the MLA cache.

## Managing KV cache memory {#kv-cache-的显存管理}

The KV cache per token has a fixed size (2 × 28 × 8 × 128 × 2 bytes = 112 KB for Qwen3-0.6B in BF16), but how many tokens a request will eventually generate is not known in advance. The naive approach reserves a contiguous block of memory for the maximum length of each request, which causes big problems:

- **Internal fragmentation**: reserve 32K, use only 500, and the rest is wasted;
- **External fragmentation**: as requests come and go, memory is cut into holes of different sizes;
- **No sharing**: several requests with the same prefix (a system prompt) each store their own copy.

**PagedAttention**, proposed by vLLM, borrows virtual memory from operating systems: split the KV cache into fixed-size **blocks** (say, 16 tokens each), and give each request a **block table** recording which physical blocks hold its logical blocks; blocks are allocated on demand and reclaimed when done, and identical prefixes can map to the same physical blocks (copy-on-write). Memory waste drops from 60–80% to under 4% (figures from the vLLM paper), so the same memory holds many more concurrent requests. The price is that the attention kernel must address memory indirectly through the block table; see [paged decode attention](cuda://advanced/attention/#一个分页-decode-attention-kernel支持-gqa) in the CUDA book.

Building on this, other common KV cache management techniques:

| Technique | What it does |
| --- | --- |
| Prefix caching | keep the KV after a request finishes, indexed by content hash (vLLM) or a radix tree (SGLang's RadixAttention), so new requests that hit the same prefix reuse it directly |
| Preemption | when memory runs short, pause some requests and drop their KV (recomputing it later) or swap it out to CPU memory |
| KV offloading | put rarely used KV in CPU memory or on SSD (LMCache, Mooncake and others) and bring it back when needed |
| KV quantization | store KV in FP8 or lower precision, doubling capacity and halving reads |

!!! interview "In an interview"
    Answer along "why it can be cached → what it saves → what problems it brings": under causal attention past tokens' K and V never change, and Q is only needed for the current token; the cache brings the computation of generating n tokens from quadratic down to linear. The price is memory: KV per token = 2 × layers × KV heads × head dimension × bytes (112 KB for Qwen3-0.6B), which sets the limits on concurrency and context. Prefill is compute intensive (it sets TTFT) and decode memory intensive (it sets TPOT); batching amortizes the weight reads (in this chapter, 8 requests decoding together take only 60% more time) but not each request's own KV. Finish with PagedAttention, prefix caching and KV quantization.

## Exercises {#练习}

**1. Compute.** Generating 1000 tokens (with a 100-token prompt), how many tokens of forward computation are needed without a KV cache, and with one?

??? success "Answer"
    Without a cache: step i processes 100 + i − 1 tokens, $\sum_{i=1}^{1000}(99 + i) = 99 \times 1000 + 500500 = 599500$ in total. With a cache: a prefill of 100, then 999 steps of 1 each (the last generated token needs no further computation), 1099 in total, about 1/545 of the former.

    ```python
    assert sum(99 + i for i in range(1, 1001)) == 599_500
    ```

**2. Estimate.** Serve LLaMA-3-8B in BF16 on an H100 (3.35 TB/s), with about 16 GB of weights and 128 KB of KV cache per token. With batch = 32 and every request at a context length of 4096, how much data must each decode step read at least? What is the latency lower bound? What share is reading KV?

??? success "Answer"
    KV cache: 32 × 4096 × 128 KB = 16 GB, as large as the weights. Each step reads at least 32 GB, for a latency lower bound of about 32 GB / 3.35 TB/s ≈ 10 ms, half of it reading KV. At a 16K context, the KV would be 4 times the weights. This shows that with long contexts and large batches, reading the KV cache is the main cost of decode.

    ```python
    kv = 32 * 4096 * 128 * 1024
    weights = 16 * 2**30
    print(f"{(kv + weights) / 3.35e12 * 1000:.1f} ms, KV 占比 {kv / (kv + weights):.0%}")
    ```

## Summary {#小结}

- [x] Causal attention guarantees that past tokens' K and V never change, so they can be cached; Q needs no caching.
- [x] The KV cache brings the computation of generation down from quadratic to linear; verify an implementation with "step-by-step computation = full recomputation".
- [x] Prefill is compute intensive (TTFT) and decode memory intensive (TPOT); during decode most of the GPU's compute sits idle.
- [x] Batching lets requests share the weight reads, the fundamental way to raise throughput; the KV cache cannot be shared this way.
- [x] PagedAttention manages the KV cache with block tables, eliminating fragmentation and enabling prefix sharing; other techniques include prefix caching, preemption, offloading and quantization.
