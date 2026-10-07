# MLA inference: two computation paths and FlashMLA

<p class="lead">The <a href="llm://transformer/attention-variants/#mla多头潜在注意力">attention variants</a> chapter of the LLM handbook covered how MLA works: the KV Cache stores only a low-dimensional latent vector, and at inference time the "decompression" matrices can be absorbed into the query and output projections. This chapter looks at it from the inference engine's side: full MLA with decoupled RoPE has two equivalent computation paths, "expand" and "absorb", whose compute differs by orders of magnitude and which suit exactly opposite scenarios; decode attention thereby turns from memory-bound to nearly compute-bound, which is why dedicated kernels like FlashMLA exist.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How many numbers does MLA's KV Cache store per token per layer? Why is the RoPE part cached separately?
    2. Of the two paths, "expand K, V" and "weight absorption", which is used for prefill and which for decode? Why?
    3. Which path should an extend with prefix caching (appending a stretch of new tokens after a long prefix) use?
    4. Why is MLA's decode attention said to be nearly compute-bound? How far is it from GQA?
    5. What problem does FlashMLA solve? Why is MLA a poor fit for tensor parallelism?

??? success "Answers (try first, then expand to compare)"
    1. 576: a 512-dimensional latent plus a 64-dimensional RoPE key shared by all heads. The RoPE rotation depends on position and sits between the up-projection and the latent, so the up-projection could not be absorbed into the query; putting the positional information in a separate 64 dimensions lets the rest be absorbed.
    2. Expand (use the up-projections to expand the cache into each head's K and V, then do ordinary multi-head attention) for prefill; absorb (fold the up-projections into the query and output projections and compute directly in latent space) for decode. Expanding costs mainly one up-projection multiply per cached token, which prefill amortizes over many new tokens; in decode there are very few new tokens and a large cache, so expanding is too expensive, while absorbing just makes each (query, key) pair about 3.4× more expensive.
    3. Choose by the number of new tokens and the prefix length: expand when there are many new tokens and the prefix is not long; when the prefix is very long, expanding the whole prefix needs a lot of memory, so expand it in chunks and merge with LSE, or use the absorb path directly when there are few new tokens.
    4. After absorption, 128 query heads share the same 576-dimensional "key", so each read of KV is reused by many heads, for an arithmetic intensity of about 240 FLOP/byte, whereas GQA's is only about 8 (each KV head is reused by one group of query heads), so it is close to compute-bound.
    5. FlashMLA is a kernel written for decode on the absorb path: it uses Tensor Cores, split-KV and paged KV to fit the shape of "MQA with a very large head dimension". MLA is a poor fit for TP: the latent is shared by all heads and cannot be split by heads, so every GPU must store a full copy of the KV.

## Two equivalent paths {#两条等价的路径}

In DeepSeek-V3's MLA, each token caches two things: a 512-dimensional latent $c$, and a 64-dimensional RoPE key $k^R$ shared by all heads. Each head's query splits into a non-positional part $q^N$ (128 dims) and a RoPE part $q^R$ (64 dims), and the attention score is

![Figure: MLA caches only the latent, and decode absorbs the up-projection into the query](../assets/figures/mla.svg){.aig-svg}

$$
s = q^{N\top} k^N + q^{R\top} k^R = q^{N\top} W_{UK}^\top c + q^{R\top} k^R
$$

So there are two algorithms:

- **Path one (expand)**: use $W_{UK}$ and $W_{UV}$ to expand each cached token's $c$ into each head's $k^N$ and $v$, append $k^R$, and do one ordinary multi-head attention (192-dimensional q/k and 128-dimensional v per head);
- **Path two (absorb)**: first move the query into latent space, $\tilde q = W_{UK} q^N$ (512 dims), so the score is $\tilde q^\top c + q^{R\top} k^R$; the weighted sum is also taken in latent space, and only at the end multiplied by $W_{UV}$ (which can also be folded into the output projection). This amounts to 128 query heads sharing the same 576-dimensional "key" and 512-dimensional "value": an MQA with a very large head dimension.

Verify with a small full MLA (including decoupled RoPE) that the two paths give the same output:

```python
import math

import torch

torch.manual_seed(0)
T, D, H = 6, 64, 4                                # tokens, hidden size, heads
NOPE, ROPE, DV, DC = 8, 4, 8, 16                  # per-head dims of the q/k non-positional part, the RoPE part, v, and the latent
W_q = torch.randn(D, H * (NOPE + ROPE)) / 8
W_dkv = torch.randn(D, DC) / 8                    # hidden → latent c (stored in the KV Cache)
W_kr = torch.randn(D, ROPE) / 8                   # hidden → the RoPE part of k shared by all heads (also cached)
W_uk = torch.randn(H, DC, NOPE) / 4               # latent → each head's non-positional part of k
W_uv = torch.randn(H, DC, DV) / 4                 # latent → each head's v
W_o = torch.randn(H * DV, D) / 8
h = torch.randn(T, D)
causal = torch.ones(T, T, dtype=torch.bool).tril()


def rope(x):                                      # x: [..., T, ROPE], rotate adjacent pairs of dims by position
    pos = torch.arange(x.shape[-2], dtype=torch.float32)[:, None]
    ang = pos / 10000 ** (torch.arange(0, ROPE, 2) / ROPE)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1).flatten(-2)


c = h @ W_dkv                                     # [T, DC]
k_r = rope(h @ W_kr)                              # [T, ROPE]
q = (h @ W_q).view(T, H, NOPE + ROPE).transpose(0, 1)                   # [H, T, NOPE+ROPE]
q_n, q_r = q[..., :NOPE], rope(q[..., NOPE:])
scale = 1 / math.sqrt(NOPE + ROPE)


def softmax(s):
    return s.masked_fill(~causal, float("-inf")).softmax(-1)


# path one (prefill): expand each head's K, V from the latent and do ordinary multi-head attention
k = torch.cat([torch.einsum("tc,hcd->htd", c, W_uk), k_r.expand(H, T, ROPE)], -1)
v = torch.einsum("tc,hcd->htd", c, W_uv)
out1 = softmax(torch.cat([q_n, q_r], -1) @ k.transpose(1, 2) * scale) @ v

# path two (decode): absorb W_uk into the query and score directly against the latent c; take the weighted sum in latent space, then multiply by W_uv at the end
q_lat = torch.einsum("htd,hcd->htc", q_n, W_uk)                          # [H, T, DC]
p = softmax((q_lat @ c.T + q_r @ k_r.T) * scale)                          # non-positional part + RoPE part
out2 = torch.einsum("hts,sc,hcd->htd", p, c, W_uv)

y1 = out1.transpose(0, 1).reshape(T, H * DV) @ W_o
y2 = out2.transpose(0, 1).reshape(T, H * DV) @ W_o
print("两条路径的输出一致：", torch.allclose(y1, y2, atol=1e-5))
print(f"每个 token 每层缓存：MHA 需要 {H * (NOPE + ROPE) + H * DV} 个数，MLA 只要 {DC + ROPE} 个（潜向量 + 共享的 RoPE 键）")
```

```text title="output"
两条路径的输出一致： True
每个 token 每层缓存：MHA 需要 80 个数，MLA 只要 20 个（潜向量 + 共享的 RoPE 键）
```

Why the RoPE part has to be split out: the rotation matrix depends on position, and in $q^\top R_m^\top R_n W_{UK} c$ the $R_n$ sits between $W_{UK}$ and $c$, so $W_{UK}$ cannot be absorbed into the query in advance. Putting the positional information in a separate 64 dimensions, with one $k^R$ shared by all heads, is what makes absorption possible. DeepSeek-V3 caches $512 + 64 = 576$ numbers per token per layer, 70 KB over 61 layers in bf16, where an MHA of the same size needs several MB.

## Which is faster: count new tokens and cached tokens {#谁快看新-token-和缓存-token-的数量}

The two paths are mathematically equivalent, but their compute is very different. With DeepSeek-V3's shapes:

```python
H, NOPE, ROPE, DV, DC = 128, 128, 64, 128, 512    # DeepSeek-V3's MLA shapes

expand = 2 * DC * H * (NOPE + DV)                 # path one: FLOPs to expand K, V for each "key token"
pair_naive = 2 * H * (NOPE + ROPE) + 2 * H * DV   # path one: QK and PV per (query, key) pair
absorb = 2 * H * NOPE * DC + 2 * H * DC * DV      # path two: absorbing W_uk and the final W_uv for each "query token"
pair_absorb = 2 * H * (DC + ROPE) + 2 * H * DC    # path two: QK and PV in latent space per (query, key) pair
print(f"每对 (query, key)：路径一 {pair_naive / 1e3:.0f}K FLOPs，路径二 {pair_absorb / 1e3:.0f}K FLOPs（{pair_absorb / pair_naive:.1f} 倍）")
print(f"路径一每个键 token 要展开 {expand / 1e6:.1f}M FLOPs；路径二每个 query token 多 {absorb / 1e6:.1f}M FLOPs")


def flops(q, k):                                   # q new tokens following k cached tokens (k includes the new tokens)
    pairs = q * k - q * (q - 1) / 2                # causal: new tokens only see what precedes them
    return k * expand + pairs * pair_naive, q * absorb + pairs * pair_absorb


for name, q, k in [("decode，上下文 8K", 1, 8192), ("prefill 8K", 8192, 8192),
                   ("在 32K 缓存前缀后追加 64 个 token", 64, 32768 + 64), ("在 32K 缓存前缀后追加 512 个 token", 512, 32768 + 512)]:
    a, b = flops(q, k)
    print(f"{name}：路径一 {a / 1e9:,.1f} GFLOPs，路径二 {b / 1e9:,.1f} GFLOPs → 用{'路径一（展开）' if a < b else '路径二（吸收）'}")
print(f"追加 q 个新 token 时，q 超过约 {expand / (pair_absorb - pair_naive):.0f} 就该改用展开的路径（k 远大于 q 时）")

kv_bytes = (DC + ROPE) * 2                         # bytes read per cached token in decode (bf16)
print(f"decode 算术强度：MLA {pair_absorb / kv_bytes:.0f} FLOP/字节；对比 GQA（64 个 q 头、8 个 KV 头、头维 128）"
      f"{2 * 64 * 128 * 2 / (8 * 128 * 2 * 2):.0f} FLOP/字节；H800 的屋脊点约 {989e12 / 3.35e12:.0f}")
```

```text title="output"
每对 (query, key)：路径一 82K FLOPs，路径二 279K FLOPs（3.4 倍）
路径一每个键 token 要展开 33.6M FLOPs；路径二每个 query token 多 33.6M FLOPs
decode，上下文 8K：路径一 275.5 GFLOPs，路径二 2.3 GFLOPs → 用路径二（吸收）
prefill 8K：路径一 3,024.0 GFLOPs，路径二 9,621.9 GFLOPs → 用路径一（展开）
在 32K 缓存前缀后追加 64 个 token：路径一 1,273.6 GFLOPs，路径二 586.8 GFLOPs → 用路径二（吸收）
在 32K 缓存前缀后追加 512 个 token：路径一 2,501.8 GFLOPs，路径二 4,726.7 GFLOPs → 用路径一（展开）
追加 q 个新 token 时，q 超过约 171 就该改用展开的路径（k 远大于 q 时）
decode 算术强度：MLA 242 FLOP/字节；对比 GQA（64 个 q 头、8 个 KV 头、头维 128）8 FLOP/字节；H800 的屋脊点约 295
```

The two paths have opposite cost structures:

- **Expand** pays its fixed cost "per cached token" (expanding $c$ into K and V for 128 heads, 33.6 million FLOPs), and each (query, key) pair is cheap;
- **Absorb** pays its fixed cost "per new token", and each (query, key) pair is 3.4× more expensive (dot products in the 576 / 512-dimensional latent space).

So decode (1 new token, tens of thousands of cached tokens) must absorb: expanding would re-expand the whole context every step, over a hundred times more expensive; prefill (as many new tokens as cached ones) uses expand, computing as ordinary multi-head attention with only a third of absorb's compute. An extend with prefix caching falls in between, with the crossover around "new tokens ≈ 170" (when the cache is much longer than the new tokens). Inference engines therefore implement both attention paths and choose by the batch's shape:

- **SGLang**: attention for DeepSeek models has several forward methods, `MHA`, `MLA`, `MHA_CHUNKED_KV` and others (`srt/models/deepseek_common/attention_forward_methods/`). Decode uses the absorbing `MLA`; prefill without a prefix uses `MHA`; with a prefix, `MLA` is used when the total prefix length is below a threshold (8192 by default, environment variable `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD`), and `MHA_CHUNKED_KV` when longer: expand the prefix in chunks, do attention chunk by chunk, and merge the chunks' results with LSE (the same merge as in [context parallelism](train://model/context/)), avoiding the memory of expanding the whole prefix at once;
- **vLLM**: `model_executor/layers/attention/mla_attention.py` splits a batch into decode and prefill parts; decode goes through each backend's `forward_mqa` (absorb: FlashMLA, FlashInfer, CUTLASS MLA and others), and prefill through the multi-head attention implementations under `v1/attention/backends/mla/prefill/`, likewise expanding in chunks when there is context.

## Decode attention becomes a compute problem {#decode-注意力变成了计算问题}

GQA's decode attention does only 8 FLOP of compute per byte for each cached token's K and V read (4 KB): thoroughly memory-bound, and the kernel's goal is to saturate memory bandwidth. After MLA absorption, 128 heads share the same 1152-byte latent, 242 FLOPs per byte, already close to the H800's ridge point (about 295); storing KV in FP8 doubles the compute per byte again, making it compute-bound. This changes how kernels are written:

- They must use Tensor Cores: treat the 128 heads as the M dimension of a matrix multiply (like stacking "several queries" together) and the latent as the K dimension, so the compute gets used;
- Tiling must balance compute and memory access: one thread block handles all 128 heads and a stretch of KV at once, with a 576-dimensional KV block in shared memory;
- With long contexts and small batches, split along the KV dimension across several SMs (split-KV, i.e. flash-decoding), then merge the partial results of each stretch.

**FlashMLA** is DeepSeek's open-source MLA decode kernel for Hopper: paged KV (block size 64), scheduling metadata precomputed from context lengths (`get_mla_metadata`, deciding which stretches of KV each SM handles), and FP8 KV support; the official figures on H800 are close to 3000 GB/s in memory-bound configurations and hundreds of TFLOPS in compute-bound ones. Later versions added kernels for sparse attention (DeepSeek-V3.2's DSA; see [MTP and sparse attention](mtp-sparse.md)). FlashInfer and CUTLASS also provide MLA decode kernels, and both SGLang and vLLM can choose among them.

## MLA and parallelism {#mla-与并行方式}

MLA's latent is one copy shared by all heads. When tensor parallelism splits attention by heads, every GPU must keep the full latent KV Cache: TP=8 means 8 identical caches, giving back the memory MLA saved. So the mainstream deployment of DeepSeek-style models is **data parallelism for attention** (DP Attention: each GPU handles different requests and stores only its own requests' KV), with expert parallelism for the MoE part (see [expert parallelism and DP Attention](../distributed/expert-parallel.md)). Another option is to limit attention's TP to a small range (say TP=2–4) and accept the KV duplication; the DeepSeek-V3 paper used TP4 attention (with sequence parallelism) in the prefill phase.

!!! interview "In an interview"
    When asked about MLA's inference implementation, answer in three layers: **cache** (576 numbers per token per layer, with the 64-dimensional RoPE key stored separately because RoPE blocks absorption); **two paths** (decode absorbs, prefill expands; give numbers like "3.4× per pair, 33.6 million FLOPs to expand each token", and explain that extend must choose by shape, with SGLang using a prefix-length threshold and chunked expansion); **kernels and parallelism** (decode arithmetic intensity is about 240, close to compute-bound, so FlashMLA uses Tensor Cores and split-KV; the latent cannot be split by heads, so TP duplicates the KV, hence DP Attention).

## Exercises {#练习}

**1. Why is each (query, key) pair more expensive after absorption?** Use the dimensions to explain where path two's 279K FLOPs per pair and path one's 82K come from.

??? success "Answer"
    Path one, per head: the QK dot product over $128 + 64 = 192$ dims and PV over 128 dims, $128 \times (192 + 128) \times 2 \approx 82$K. Path two, per head: QK over the latent space's $512 + 64 = 576$ dims and PV over 512 dims, $128 \times (576 + 512) \times 2 \approx 279$K. Absorption trades each head's "small-dimension dot products" for "large-dimension dot products" in exchange for not expanding the cache: each pair costs more, but the 33.6 million FLOPs of expansion per cached token are saved.

**2. An FP8 KV Cache.** With MLA's KV stored in FP8, what is the arithmetic intensity of decode attention? What does it mean for kernel design?

??? success "Answer"
    Each cached token reads 576 bytes (not counting scales), for an arithmetic intensity of about $278528 / 576 \approx 484$ FLOP/byte, beyond the H800's BF16 ridge point (about 295), so it becomes compute-bound. The kernel's bottleneck then shifts from "reading KV" to "Tensor Core throughput": either use FP8 Tensor Cores for QK (with careful handling of precision), or accept being compute-bound and focus optimization on Tensor Core utilization. For the system, FP8 KV fits twice the context in the same memory, but decode latency does not fall proportionally: the bottleneck is no longer bandwidth.

**3. Why might the absorb path also help prefill with long contexts?**

??? success "Approach"
    The expand path must materialize K and V for the whole context: 128 heads at 192 + 128 dims each, about 80 KB per token (bf16), so a 32K prefix is 2.6 GB, for one layer alone. When memory is tight, either expand in chunks (SGLang's `MHA_CHUNKED_KV`, vLLM's chunked context) and merge with LSE, or use the absorb path directly when there are few new tokens. So real systems choose not only by FLOPs but also by memory and kernel availability.

## Summary {#小结}

- [x] MLA caches a 512-dimensional latent + a 64-dimensional shared RoPE key per token per layer; RoPE must be decoupled for absorption to work.
- [x] Two equivalent paths: expand (large fixed cost per cached token, cheap per pair) for prefill; absorb (fixed cost per new token, 3.4× more expensive per pair) for decode; extend chooses by new-token count and prefix length, expanding long prefixes in chunks and merging with LSE.
- [x] After absorption, decode attention's arithmetic intensity is about 240 FLOP/byte (GQA about 8), close to compute-bound; kernels such as FlashMLA adapt with Tensor Cores, split-KV and paged KV.
- [x] The latent cannot be split by heads, and TP duplicates the KV Cache; the mainstream deployment is DP Attention + EP.
