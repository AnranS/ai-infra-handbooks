# Long context, KV eviction and sparse attention

<p class="lead">As context grows from 4K to 128K and 1M, inference's bottleneck shifts with it: prefill attention compute grows with the square of the length, the KV Cache grows linearly with it, and every decode step reads the entire KV. This chapter first works out at what length the cost of long context becomes significant, then runs two experiments on a real model: StreamingLLM-style KV eviction (keeping only attention sinks and a recent window), and a comparison of the errors of several sparse attention selection strategies.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. At what context length does attention compute start to exceed the linear layers?
    2. Why does keeping only the KV of the most recent N tokens collapse? What fix did StreamingLLM make?
    3. What is the essential difference between static KV eviction and dynamic sparse attention (such as DeepSeek's DSA)?
    4. What dedicated support do inference engines need for long contexts?

??? success "Answers (try first, then expand to compare)"
    1. Attention compute per layer is about $4 \cdot d \cdot$ context length, and the linear layers about $2 \times$ the parameter count; by this chapter's estimate, attention compute catches up with the linear layers at about 38K for Qwen2.5-7B and about 54K for LLaMA-3-70B.
    2. The model puts a lot of attention on the first few tokens (attention sinks); dropping them scrambles the softmax distribution, and the output collapses. StreamingLLM always keeps the KV of the first few tokens in addition to the recent window, so the model handles long input streams stably (though information outside the window is still lost).
    3. Static eviction permanently drops KV by some rule, never to be used again; dynamic sparse attention keeps all the KV, and each query uses a trained lightweight scorer to pick a small subset to attend to (DSA picks 2048 per query). Different queries can look at different tokens, so no information is lost; only compute and bandwidth are saved.
    4. KV management for sliding-window / mixed local-global attention, attention sinks, kernels for sparse attention and caching for its indexer, chunked prefill, context parallelism (very long prompts), plus KV offloading and capacity planning.

<!-- comic ../assets/comics/long-context.webp is in Chinese; put it back once the English version exists -->

## The cost of long context {#长上下文的代价}

Each token costs about 2 × parameter count in the linear layers, while attention compute is proportional to the context length the token sees: $4 \cdot S \cdot n_h \cdot d_h$ per layer (half for $QK^\top$, half for $PV$). Where they are equal:

```python
models = {"Qwen2.5-7B": (7.6e9, 28, 28, 128), "LLaMA-3-70B": (70.6e9, 80, 64, 128)}
for name, (params, layers, heads, head_dim) in models.items():
    crossover = 2 * params / (4 * layers * heads * head_dim)          # context length where attention = linear layers
    kv = 2 * layers * {"Qwen2.5-7B": 4, "LLaMA-3-70B": 8}[name] * head_dim * 2
    print(f"{name}：上下文约 {crossover / 1000:.0f}K 时注意力计算量追上线性层；"
          f"128K 上下文的 KV Cache 为 {kv * 131072 / 1e9:.1f} GB")
```

```text title="输出"
Qwen2.5-7B：上下文约 38K 时注意力计算量追上线性层；128K 上下文的 KV Cache 为 7.5 GB
LLaMA-3-70B：上下文约 54K 时注意力计算量追上线性层；128K 上下文的 KV Cache 为 42.9 GB
```

In other words, within a few tens of thousands of tokens attention is not yet the bulk of the compute; beyond 128K most of prefill's time goes to attention; a single 128K request's KV is several to tens of GB, and a few such requests together exceed the model's weights. Decode is more direct: every step reads the entire KV, so TPOT grows linearly with context. The responses fall into three groups:

1. **Less KV**: at the architecture level (GQA/MLA, mixing sliding-window and global attention, linear attention; see the LLM handbook's [model tour](llm://synthesis/models/)), or evicting unimportant KV at runtime;
2. **Look at only part of the KV each step**: sparse attention;
3. **Spread over more GPUs**: context parallelism ([previous part](../distributed/pp-cp.md#上下文并行)), KV offloading ([tiered caching](../distributed/kv-offload.md)).

## KV eviction: StreamingLLM {#kv-淘汰streamingllm}

The simplest eviction policy keeps only the most recent N tokens (a sliding window). On a 1600-token text, process it in chunks of 64 tokens, evicting the KV down to 256 after each chunk, and look at perplexity:

```python
import math
import re
import torch
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer, apply_rope
from tree_spec import forward_with_mask

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
raw = open("../llm/docs/assets/sample-passage.txt").read()   # frozen sample text: a snapshot of the body of the LLM handbook's "language models" chapter
ids = tok(re.sub(r"[#*`>|\-\[\]()!]", "", raw)).input_ids[:1600]    # the body of a chapter from the LLM handbook
L = model.cfg.num_hidden_layers

def streaming_perplexity(policy, budget=256, sinks=4, chunk=64):
    cache, nll, count = KVCache(L), 0.0, 0
    for start in range(0, len(ids) - 1, chunk):
        piece = ids[start:start + chunk]
        causal = torch.ones(len(piece), len(piece), dtype=torch.bool).tril()
        # positions are the tokens' true positions in the text; K in the cache was already rotated by its own position when written
        logits, new_kv = forward_with_mask(model, torch.tensor(piece), torch.arange(start, start + len(piece)),
                                           cache, causal)
        target = ids[start + 1:start + len(piece) + 1]
        nll += torch.nn.functional.cross_entropy(logits[:len(target)], torch.tensor(target), reduction="sum").item()
        count += len(target)
        for layer, (k, v) in enumerate(new_kv):
            cache.update(layer, k, v)
        S = cache.k[0].shape[2]
        if policy != "全部保留" and S > budget:
            if policy == "只保留最近 256 个":
                keep = list(range(S - budget, S))
            else:                                                     # sinks at the start + the recent window
                keep = list(range(sinks)) + list(range(S - (budget - sinks), S))
            for layer in range(L):
                cache.k[layer], cache.v[layer] = cache.k[layer][:, :, keep], cache.v[layer][:, :, keep]
    return math.exp(nll / count)

for policy in ("全部保留", "只保留最近 256 个", "开头 4 个 + 最近窗口"):
    print(f"{policy:14s} 困惑度 {streaming_perplexity(policy):7.2f}")
```

```text title="输出"
全部保留           困惑度   27.12
只保留最近 256 个    困惑度  677.04
开头 4 个 + 最近窗口  困惑度   35.82
```

Keeping only the recent window sends perplexity from 27.1 to 677, a complete collapse of the model; while **additionally keeping the first 4 tokens** brings perplexity back to 35.8. This is exactly what StreamingLLM (Xiao et al., 2023) found: the model puts a lot of attention on the very first few tokens, treating them as a trash bin for "attend to nothing" (the LLM handbook measured on the same model that from layer 6 on, 40%–77% of attention lands on the first token; see [attention sinks](llm://transformer/attention/#真实模型里的注意力注意力汇聚)). Evict them, and the softmax denominator suddenly loses a large chunk, distorting the distribution of all the remaining attention. Keep these few "sinks", and a fixed-size KV can handle an endlessly long input stream.

StreamingLLM solves "not collapsing", not "not losing information": content outside the window is still discarded, and tasks that need to recall distant details still fail. Finer eviction policies decide which tokens to keep by attention scores: H2O keeps the "heavy hitter" tokens with the highest accumulated attention, and SnapKV, at the end of prefill, picks the KV each head should keep from the attention of the last stretch of queries.

Visualize the eviction policy: how the budget and the number of sink tokens decide who stays in the cache:

<div class="aig-widget" data-widget="kv-evict"></div>

## Sparse attention: choosing dynamically at every step {#稀疏注意力每一步动态选择}

Eviction is **permanent**: dropped KV never comes back. Sparse attention keeps all the KV (in GPU memory, or even offloaded to CPU), but at each step lets a query attend to only a small part of it, and **each query can pick a different part**. The question becomes: how to pick the most important part both quickly and accurately?

Using layer 12's Q, K and V from a real model, compare the error against full attention of several strategies where "each query looks at only 256 tokens" (computed for the last 64 queries):

```python
LAYER, budget = 12, 256
captured = {}
hook = model.layers[LAYER].self_attn.register_forward_pre_hook(
    lambda mod, args: captured.update(x=args[0], cos=args[1], sin=args[2]))
cache = KVCache(L)
with torch.no_grad():
    model(torch.tensor([ids]), cache)
hook.remove()
attn = model.layers[LAYER].self_attn
with torch.no_grad():
    q = apply_rope(attn.q_norm(attn.q_proj(captured["x"]).view(1, -1, attn.nh, attn.hd).transpose(1, 2)),
                   captured["cos"], captured["sin"])[0]                       # [heads, S, D]
rep = attn.nh // attn.nkv
K, V = cache.k[LAYER][0].repeat_interleave(rep, 0), cache.v[LAYER][0].repeat_interleave(rep, 0)
S = K.shape[1]
qpos, kpos = torch.arange(S - 64, S), torch.arange(S)
causal = qpos[:, None] >= kpos[None, :]
scores = (q[:, qpos] @ K.transpose(-1, -2) * attn.hd ** -0.5).masked_fill(~causal, float("-inf"))
full = scores.softmax(-1) @ V

def error_with(selected):
    out = scores.masked_fill(~selected, float("-inf")).softmax(-1) @ V
    return ((out - full).norm() / full.norm()).item()

window = (kpos[None, None] > qpos[:, None] - budget) & causal
sink_window = ((kpos[None, None] > qpos[:, None] - (budget - 4)) | (kpos[None, None] < 4)) & causal
oracle = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, scores.topk(budget, -1).indices, True) & causal
page = 16                                                              # Quest: estimate an upper bound on attention scores per page
pages = K[:, : S // page * page].view(K.shape[0], S // page, page, -1)
kmin, kmax = pages.amin(2), pages.amax(2)
upper = torch.maximum(q[:, qpos, None, :] * kmin[:, None], q[:, qpos, None, :] * kmax[:, None]).sum(-1)
chosen = torch.zeros_like(upper, dtype=torch.bool).scatter_(-1, upper.topk(budget // page, -1).indices, True)
quest = torch.cat([chosen.repeat_interleave(page, -1),
                   torch.ones(*chosen.shape[:2], S - S // page * page, dtype=torch.bool)], -1) & causal
for name, sel in [("最近 256 个（滑动窗口）", window), ("开头 4 个 + 最近窗口", sink_window),
                  ("按页估计上界选 16 页（Quest）", quest), ("每个 query 精确的 top-256", oracle)]:
    print(f"{name:22s} 相对误差 {error_with(sel):.3f}")
```

```text title="输出"
最近 256 个（滑动窗口）         相对误差 0.705
开头 4 个 + 最近窗口          相对误差 0.312
按页估计上界选 16 页（Quest）    相对误差 0.232
每个 query 精确的 top-256   相对误差 0.094
```

- The static window strategy has the largest error, which improves greatly once sinks are added;
- **Picking exactly the 256 highest-scoring tokens for each query** gives an error of only 9.4%: this is what "choosing right" can achieve, showing that attention really is highly sparse;
- But exact selection requires computing all the scores first, which saves no compute. Quest estimates an upper bound on scores from each page's minimum and maximum keys at very little cost; here it does somewhat better than "sinks + window", but is still well short of exact selection.

Real systems approach exact selection with a **fast and accurate scorer**. DeepSeek-V3.2's DSA (DeepSeek Sparse Attention) trains a lightweight "lightning indexer" that quickly scores all past tokens with very few heads in FP8, and each query does full attention only over the 2048 highest-scoring tokens; NSA (Native Sparse Attention) combines three branches: compressed coarse-grained attention, selected fine-grained blocks, and a sliding window. Both have the model adapt to sparse attention during training, rather than approximating after the fact at inference time.

## Support in inference engines {#推理引擎中的支持}

- **Sliding windows and mixed attention**: vLLM's `SlidingWindowManager` keeps only the blocks within the window and frees those outside promptly; different layers of a hybrid model use different KV managers (KV Cache groups; see [the vLLM walkthrough](../source/vllm.md#主线三调度器与-kv-cache-管理));
- **Attention sinks**: models like gpt-oss with explicit sink terms need support from the attention backend (vLLM's `SinkFullAttentionManager`, SGLang's corresponding backends);
- **Sparse attention**: DeepSeek-V3.2's DSA needs dedicated indexer kernels and the indexer's own KV cache (the sparse MLA backends and `sparse_attn_indexer` in vLLM, and `nsa_backend.py` and `dsa_backend.py` in SGLang);
- **Prefill of long prompts**: chunked prefill is a must, or one 128K request stalls the whole service; for even longer ones, use context parallelism;
- **Position encoding extension**: beyond the training length, RoPE scaling such as YaRN extends it (set in `rope_scaling` in the model config), and inference engines must implement the corresponding RoPE variants.

!!! source "Source code"
    - **vLLM**: the KV managers are in `vllm/v1/core/single_type_kv_cache_manager.py` (`SlidingWindowManager`, `ChunkedLocalAttentionManager`, `SinkFullAttentionManager` and others); sparse attention lives in `vllm/model_executor/layers/sparse_attn_indexer.py`, plus sparse MLA backends such as `flashmla_sparse.py` and `flashinfer_mla_sparse.py` and `indexer.py` under `vllm/v1/attention/backends/mla/`; `vllm/v1/hisparse/` is tiered KV management for sparse attention.
    - **SGLang**: `srt/layers/attention/nsa_backend.py`, `dsa_backend.py` and `dsa/`, `srt/mem_cache/sparsity/` and `hisparse_memory_pool.py`; for sliding windows, `swa_memory_pool.py` and `swa_radix_cache.py`.

!!! interview "In an interview"
    "How do you support a 1M context?" can be answered in three layers: **model architecture** (GQA/MLA, mixed local and global attention, linear attention, native sparse attention) → **inference system** (chunked prefill, context parallelism, KV offloading, indexing and selection for sparse attention) → **lossy approximation** (KV eviction: StreamingLLM keeping sinks, H2O/SnapKV keeping by importance, and their costs). Using this chapter's experiments to show "keeping only the window collapses, keeping sinks does not" and "attention is highly sparse, but only accurate selection helps" is very convincing.

## Exercises {#练习}

**1. How many sinks to keep?** Change `sinks` from 4 to 1 and 16. How does perplexity change? Why might keeping 1 be enough?

??? success "Approach"
    Just change the parameter of `streaming_perplexity` and run it. Most models' attention sinks concentrate on the first token (in the LLM handbook's measurements, token 0 absorbs most of the sink attention), so keeping 1 recovers most of the effect; keeping 4 is safer (the StreamingLLM paper's default), and more brings little gain while taking slots from the window.

**2. Why can't sparse attention be used directly on every model?** If exact top-256 has an error of only 9.4%, why not use top-k sparse attention on any model at inference time?

??? success "Answer"
    - Exact top-k must compute all the scores first, so compute is not reduced; only the softmax and PV parts are saved. Saving compute for real requires approximate scoring (Quest, indexers), and when approximate scoring picks badly the error is much larger;
    - Errors accumulate across layers and tokens; a per-layer error around 9% spread over 28 layers and thousands of tokens can noticeably degrade quality, especially on tasks that need precise retrieval of distant information ("needle in a haystack");
    - That is why DSA and NSA both introduce sparse attention during training, teaching the model to work under sparsity, rather than swapping it in after the fact at inference time.

## Summary {#小结}

- [x] Attention compute catches up with the linear layers at a few tens of thousands of tokens; a few long-context requests' KV can exceed the weights, and decode TPOT grows linearly with context.
- [x] Keeping only the recent window's KV collapses the model; keeping the attention sinks at the start (StreamingLLM) handles long input streams stably, but information outside the window is still lost.
- [x] Attention is highly sparse, and exact selection can attend to a small part of the KV with little error; practical sparse attention (DSA, NSA) approaches exact selection with trained lightweight scorers.
- [x] Engines need to support KV management for sliding-window/mixed attention, attention sinks, sparse attention kernels, chunked prefill and context parallelism.
