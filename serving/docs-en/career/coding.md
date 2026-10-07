# Implementation problems

<p class="lead">Interviews for inference roles almost always include writing code live, and the problems cluster into three kinds: writing a model's core components in PyTorch (attention, RoPE, RMSNorm, sampling), writing the data structures and algorithms of an inference system (LRU, prefix trees, scheduling, all-reduce), and writing GPU kernels (softmax, reductions, GEMM). This chapter gives the most frequently asked problems of the first two kinds: each starts with what the interviewer is looking for, then gives an answer you can run directly and check item by item against a reference implementation. GPU kernel problems are in the CUDA handbook's <a href="cuda://career/interview/">interview question bank</a> and its per-operator chapters.</p>

!!! tip "How to practice"
    First write it yourself on paper or in an empty editor under time pressure (15–20 minutes per problem), then compare with the answer and run its test code. In an interview, after finishing, proactively state the complexity, the edge cases and how you would test it; that often earns more than the code itself.

## 1. GQA attention with a KV Cache {#1-带-kv-cache-的-gqa-注意力}

**Problem**: implement a multi-head attention module supporting GQA (fewer KV heads than query heads), a causal mask, and a KV Cache for incremental decoding.

**What it tests**: shape manipulation (`view`/`transpose`), the scaling factor, head replication for GQA, the offset of the causal mask when there is a cache, and appending to the cache.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.nh, self.nkv, self.hd = n_heads, n_kv_heads, dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, dim, bias=False)

    def forward(self, x, cache=None):
        """x: [B, T, dim]；cache: None 或 (k, v)，形状 [B, n_kv, S_past, hd]。返回 (输出, 新的 cache)。"""
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)        # [B, nh, T, hd]
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)       # [B, nkv, T, hd]
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        S = k.shape[2]
        rep = self.nh // self.nkv
        kr, vr = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = q @ kr.transpose(-2, -1) / math.sqrt(self.hd)             # [B, nh, T, S]
        mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)      # the new tokens sit at the end of the sequence
        scores = scores.masked_fill(~mask, float("-inf"))
        out = scores.softmax(-1) @ vr                                       # [B, nh, T, hd]
        return self.wo(out.transpose(1, 2).reshape(B, T, -1)), (k, v)


torch.manual_seed(0)
attn = GQAttention(dim=64, n_heads=8, n_kv_heads=2)
x = torch.randn(2, 10, 64)
full, _ = attn(x)                                                           # compute the whole sequence at once
out, cache = attn(x[:, :6])                                                 # prefill 6 tokens first
steps = [out]
for t in range(6, 10):                                                      # then decode one by one
    o, cache = attn(x[:, t:t + 1], cache)
    steps.append(o)
print("增量计算与一次计算的最大误差：", (torch.cat(steps, 1) - full).abs().max().item())
q = attn.wq(x).view(2, 10, 8, 8).transpose(1, 2)                            # compare with PyTorch's official implementation
k = attn.wk(x).view(2, 10, 2, 8).transpose(1, 2).repeat_interleave(4, 1)
v = attn.wv(x).view(2, 10, 2, 8).transpose(1, 2).repeat_interleave(4, 1)
ref = attn.wo(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 10, 64))
assert torch.allclose(full, ref, atol=1e-5) and torch.allclose(torch.cat(steps, 1), full, atol=1e-5)
```

<!-- i18n:diagram a0111d85fe -->
```text
max difference between incremental and one-shot computation: 1.1920928955078125e-07
```

**Common mistakes**: writing the mask as `tril()` without an offset (with a cache, decode queries see nothing or see the wrong positions); using `repeat` instead of `repeat_interleave` for GQA (scrambling the head correspondence); forgetting to divide by $\sqrt{d_h}$.

## 2. RoPE {#2-rope}

**Problem**: implement rotary position embeddings and explain why they encode relative positions.

**What it tests**: the frequencies $\theta^{-2i/d}$, the two equivalent formulations (pairing adjacent dimensions, or pairing the two halves), and positions starting at the cache length.

```python
def rope(x, positions, theta=10000.0):
    """x: [..., T, d]，把第 i 维与第 i + d/2 维看成一个二维向量，按 位置 × 频率 旋转。"""
    d = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, d, 2).float() / d)               # [d/2]
    angles = positions.float()[:, None] * inv_freq[None, :]                 # [T, d/2]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

from mini_llm import apply_rope, rope_cos_sin
x = torch.randn(2, 4, 16, 64)
pos = torch.arange(100, 116)                                                # with a cache, positions start at the cache length
cos, sin = rope_cos_sin(pos, 64, 10000.0)
print("与 mini_llm 的实现一致：", torch.allclose(rope(x, pos), apply_rope(x, cos, sin), atol=1e-5))

q, k = torch.randn(64), torch.randn(64)                                     # relative positions: it depends only on the difference of positions
s1 = rope(q[None], torch.tensor([10])) @ rope(k[None], torch.tensor([7])).T
s2 = rope(q[None], torch.tensor([110])) @ rope(k[None], torch.tensor([107])).T
print("位置 (10, 7) 与 (110, 107) 的注意力分数之差：", (s1 - s2).abs().item())
assert torch.allclose(rope(x, pos), apply_rope(x, cos, sin), atol=1e-5) and (s1 - s2).abs() < 1e-3
```

<!-- i18n:diagram 3dbb3babb9 -->
```text
matches mini_llm's implementation: True
difference between the attention scores at (10, 7) and (110, 107): 7.152557373046875e-06
```

## 3. RMSNorm {#3-rmsnorm}

**Problem**: implement RMSNorm and explain why inference engines fuse it with the residual add.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()                                                     # compute the statistics in FP32 to avoid BF16 overflow and precision loss
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight

norm = RMSNorm(64)
nn.init.normal_(norm.weight)
x = torch.randn(4, 64)
ref = F.rms_norm(x, (64,), norm.weight, eps=1e-6)
print("与 torch.nn.functional.rms_norm 的最大误差：", (norm(x) - ref).abs().max().item())
assert torch.allclose(norm(x), ref, atol=1e-6)
```

<!-- i18n:diagram 52ebd632b4 -->
```text
max difference from torch.nn.functional.rms_norm: 0.0
```

**Key point**: RMSNorm is a memory-bound operator (one read, one write, very little compute). In inference, if `residual = x + residual; y = rmsnorm(residual)` is split into two kernels, the whole tensor is read and written one extra time; fused, it is read and written once, which is the fused add RMSNorm in vLLM and SGLang (see the CUDA handbook's [softmax and normalization](cuda://kernels/softmax-norm/)).

## 4. Online softmax and blocked attention (the core of FlashAttention) {#4-在线-softmax-与分块注意力flashattention-的核心}

**Problem**: compute attention in blocks without materializing the full attention matrix.

**What it tests**: the recurrence of online softmax (maintaining the row maximum $m$ and the denominator $l$, rescaling earlier results when a larger value appears), and dividing by the denominator at the end.

```python
def flash_attention(q, k, v, block=16):
    """q: [Tq, d]，k, v: [Tk, d]（单头、无掩码）。每次只处理一块 K/V，显存占用与序列长度无关。"""
    scale = q.shape[-1] ** -0.5
    m = torch.full((q.shape[0], 1), float("-inf"))                         # the largest score each row has seen so far
    l = torch.zeros(q.shape[0], 1)                                          # each row's current softmax denominator (relative to m)
    acc = torch.zeros_like(q)                                               # the unnormalized output
    for start in range(0, k.shape[0], block):
        s = q @ k[start:start + block].T * scale                            # [Tq, block]
        m_new = torch.maximum(m, s.max(-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        correction = torch.exp(m - m_new)                                   # rescale the old accumulation to the new maximum
        l = l * correction + p.sum(-1, keepdim=True)
        acc = acc * correction + p @ v[start:start + block]
        m = m_new
    return acc / l

q, k, v = torch.randn(32, 64), torch.randn(100, 64), torch.randn(100, 64)
ref = torch.softmax(q @ k.T / 8, -1) @ v
print("分块计算与标准注意力的最大误差：", (flash_attention(q, k, v) - ref).abs().max().item())
assert torch.allclose(flash_attention(q, k, v), ref, atol=1e-5)
```

<!-- i18n:diagram 7ba838a27b -->
```text
max difference between blocked and standard attention: 2.384185791015625e-07
```

**Follow-ups**: how to add a causal mask (skip blocks entirely in the future, mask the blocks on the diagonal)? Why save $m$ and $l$ (or the log-sum-exp) for the backward pass? In decode the query is a single row, so how do you parallelize (Flash-Decoding: split along K/V across thread blocks and merge with LSE at the end; see [context parallelism](../distributed/pp-cp.md#ring-attention)).

## 5. top-p sampling {#5-top-p-采样}

**Problem**: implement temperature + top-p sampling, done for a whole batch at once.

```python
def sample_top_p(logits, temperature, top_p, generator=None):
    """logits: [B, V]；temperature, top_p: [B]。"""
    probs = (logits / temperature[:, None]).softmax(-1)
    sorted_probs, idx = probs.sort(-1, descending=True)
    remove = (sorted_probs.cumsum(-1) - sorted_probs) >= top_p[:, None]    # the cumulative probability ahead already reaches top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    choice = torch.multinomial(sorted_probs / sorted_probs.sum(-1, keepdim=True), 1, generator=generator)
    return idx.gather(-1, choice).squeeze(-1)

logits = torch.tensor([[2.0, 1.0, 0.5, -1.0, -3.0]])
g = torch.Generator().manual_seed(0)
draws = torch.stack([sample_top_p(logits, torch.tensor([1.0]), torch.tensor([0.8]), g) for _ in range(20000)])
freq = torch.bincount(draws.flatten(), minlength=5) / 20000
p = logits.softmax(-1)[0]
kept = p[:2] / p[:2].sum()                                                  # the first two tokens' cumulative probability just exceeds 0.8
print("采样频率：", [round(f, 3) for f in freq.tolist()], " 理论值：", [round(x, 3) for x in kept.tolist()] + [0, 0, 0])
assert (freq[:2] - kept).abs().max() < 0.02 and freq[2:].sum() == 0
```

<!-- i18n:diagram f0734755a9 -->
```text
sampling frequencies: [0.733, 0.267, 0.0, 0.0, 0.0]  theoretical: [0.731, 0.269, 0, 0, 0]
```

**Key points**: the condition is "the cumulative probability ahead of it ≥ p", which guarantees at least one token is kept; the batched implementation must avoid Python loops; in follow-ups you can mention that vLLM replaces `multinomial` with an exponential race to avoid CPU-GPU synchronization ([the sampler chapter](../engine/sampler-api.md#批量采样)).

## 6. An O(1) LRU cache {#6-o1-的-lru-缓存}

**Problem**: implement an LRU cache where both `get` and `put` are O(1) (used by prefix caches, expert caches and KV offloading).

**What it tests**: a hash map + a doubly linked list; moving a node to the head on access; removing from the tail on eviction. Interviews usually require writing the linked list by hand rather than using `OrderedDict`.

```python
class Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key, self.value, self.prev, self.next = key, value, None, None


class LRUCache:
    def __init__(self, capacity):
        self.capacity, self.map = capacity, {}
        self.head, self.tail = Node(), Node()                               # sentinels: newest after head, oldest before tail
        self.head.next, self.tail.prev = self.tail, self.head

    def _unlink(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def _push_front(self, node):
        node.prev, node.next = self.head, self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        node = self.map.get(key)
        if node is None:
            return None
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key, value):
        if key in self.map:
            node = self.map[key]
            node.value = value
            self._unlink(node)
        else:
            if len(self.map) == self.capacity:
                oldest = self.tail.prev
                self._unlink(oldest)
                del self.map[oldest.key]
            node = self.map[key] = Node(key, value)
        self._push_front(node)

cache = LRUCache(2)
cache.put("a", 1); cache.put("b", 2); cache.get("a"); cache.put("c", 3)   # b was least recently used and got evicted
print("a:", cache.get("a"), " b:", cache.get("b"), " c:", cache.get("c"))
assert (cache.get("a"), cache.get("b"), cache.get("c")) == (1, None, 3)
```

```text
a: 1  b: None  c: 3
```

## 7. Ring all-reduce {#7-ring-all-reduce}

**Problem**: n processes each hold a vector of length N; sum them with a ring algorithm so every process ends up with the total. State the communication volume.

**What it tests**: first reduce-scatter (n−1 steps, each sending a chunk to the next process which accumulates it), then all-gather (n−1 steps, passing the complete chunks around); each process sends $2\frac{n-1}{n}N$ elements in total, almost independent of the process count, which is why the ring algorithm is bandwidth-optimal.

```python
def ring_all_reduce(vectors):
    """模拟 n 个进程：vectors[r] 是进程 r 的数据。返回每个进程最终的数据，以及每个进程发送的元素数。"""
    n = len(vectors)
    chunks = [[c.clone() for c in v.chunk(n)] for v in vectors]             # chunks[r][c]: process r's chunk c (a copy, leaving the input unchanged)
    sent = 0
    for step in range(n - 1):                                               # reduce-scatter
        outgoing = [(r, (r - step) % n, chunks[r][(r - step) % n].clone()) for r in range(n)]
        for r, c, data in outgoing:
            chunks[(r + 1) % n][c] += data                                  # send to the next process, which accumulates it
            sent += data.numel()
    # at this point process r's chunk (r + 1) % n holds the complete sum
    for step in range(n - 1):                                               # all-gather
        outgoing = [(r, (r + 1 - step) % n, chunks[r][(r + 1 - step) % n].clone()) for r in range(n)]
        for r, c, data in outgoing:
            chunks[(r + 1) % n][c] = data
            sent += data.numel()
    return [torch.cat(c) for c in chunks], sent // n

vectors = [torch.randn(12) for _ in range(4)]
total = torch.stack(vectors).sum(0)
results, per_rank = ring_all_reduce(vectors)
print("每个进程都得到了总和：", all(torch.allclose(r, total, atol=1e-6) for r in results),
      f"；每个进程发送 {per_rank} 个元素 = 2 × (n-1)/n × N = {2 * 3 / 4 * 12:.0f}")
assert all(torch.allclose(r, total, atol=1e-6) for r in results)
```

<!-- i18n:diagram da7db84752 -->
```text
every process got the sum: True ; each process sent 18 elements = 2 × (n-1)/n × N = 18
```

An easy trap: `v.chunk(n)` returns **views** of the original tensor, so without copying first, `+=` silently modifies the caller's input. This book's first version of the answer made exactly this mistake, and because the reference `total` was computed only after the call, the comparison failed inexplicably. In-place operations in PyTorch code are worth pointing out proactively in an interview.

## 8. Turning estimation problems into code {#8-估算题的代码化}

**Problem**: write a function that takes the model configuration, batch size, context length and GPU parameters, and outputs the lower bound on one decode step's time plus the maximum concurrency memory allows.

This turns the LLM handbook's [estimation methods](llm://inference/estimation/) into code, and interviews often ask for it as mental arithmetic:

```python
def decode_estimate(params, n_layers, n_kv_heads, head_dim, batch, context, bw=3.35e12, mem=80e9,
                    weight_bytes=2, kv_bytes=2, reserve=0.1):
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * kv_bytes
    step_ms = (params * weight_bytes + batch * context * kv_per_token) / bw * 1e3
    free = mem * (1 - reserve) - params * weight_bytes
    return step_ms, int(free // (context * kv_per_token))

step, max_batch = decode_estimate(8.03e9, 32, 8, 128, batch=32, context=4096)
print(f"LLaMA-3-8B，batch 32、上下文 4K：decode 一步下限 {step:.1f} ms，单卡最多约 {max_batch} 个这样的请求")
```

<!-- i18n:diagram 7e7828b56f -->
```text
LLaMA-3-8B, batch 32, context 4K: decode step floor 9.9 ms, about 104 such requests per GPU
```

## More problems {#更多题目}

| Problem | Reference |
| --- | --- |
| Paged attention (given a block table) | [the paged KV Cache](../engine/paged-kv.md) |
| A continuous batching scheduler | [the scheduler](../engine/scheduler.md) |
| Insertion, matching and eviction in a radix tree | [prefix caching](../engine/prefix-cache.md) |
| A tensor-parallel linear layer | [tensor parallelism](../distributed/tensor-parallel.md) |
| MoE routing and grouping by expert | the LLM handbook's [MoE](llm://transformer/moe/) |
| Rejection sampling for speculative decoding | the LLM handbook's [serving](llm://inference/serving/#投机解码) |
| Training and encoding with BPE tokenization | the LLM handbook's [tokenization](llm://basics/tokenization/) |
| CUDA: reductions, softmax, GEMM, transpose | the CUDA handbook's [classic operators](cuda://kernels/reduction/) and [interview question bank](cuda://career/interview/) |

## Summary {#小结}

- [x] For model-component problems the keys are shapes, mask offsets, GQA head correspondence and numerical stability; check your result against the official implementation.
- [x] Online softmax is the shared foundation of FlashAttention, Flash-Decoding and ring attention, so be able to write it from memory.
- [x] System problems often ask for LRU, prefix trees, scheduling and all-reduce; be able to state the complexity and the communication volume.
- [x] Estimation problems must become code or mental arithmetic: weight bytes + KV bytes divided by bandwidth, and memory minus weights divided by KV per request.
