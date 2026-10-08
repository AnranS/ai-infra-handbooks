# Serving multiple LoRAs: segmented matrix multiplies and adapter scheduling

<p class="lead">One base model with thousands of LoRA fine-tuned versions is the most common deployment shape in SaaS platforms and inside companies. Running a separate instance per version is too wasteful; putting them in the same instance and computing them together in one batch is the right approach, but it requires different requests in one batch to multiply by different low-rank matrices. This chapter implements two ways to compute them in batches (BGMV and SGMV), then looks at how adapters move between GPU memory, host memory and storage, and how this relates to prefix caching and routing.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What is LoRA's forward pass? What are the pros and cons of merging the weights versus not merging?
    2. When requests in one batch use different adapters, how do you compute them in one go? How do BGMV and SGMV differ?
    3. How big is one adapter? How many can stay resident in GPU memory?
    4. What must the prefix cache watch out for in multi-LoRA serving?

??? success "Answers (try first, then expand to compare)"
    1. $y = xW + s \cdot (xA)B$. Merging (adding $sAB$ into $W$) makes inference as fast as the original model, but one instance can then serve only one adapter; without merging, the base part is computed for all requests together while the LoRA part is computed per request, so one instance can serve many adapters, at the cost of the extra LoRA compute and reads.
    2. The base part is one matrix multiply for the whole batch; the LoRA part is computed with each request's own adapter. BGMV: each token gathers its own adapter's A and B by index and does a small matrix × vector per token, suited to decode; SGMV: tokens using the same adapter are grouped together and each segment does one matrix multiply, suited to prefill.
    3. About 80 MB for an 8B base at rank 16 across all 7 projections; reserving 8 GB of memory keeps about 100 resident, and host memory can cache thousands.
    4. The same prefix has different K and V under different adapters (LoRA applies to the q, k and v projections), so a block's hash key must include the adapter's id, or it will hit KV computed for another adapter.

## Not merging the weights, computing in batches {#不合并权重批量计算}

LoRA's forward pass is $y = xW + s \cdot (xA)B$, where $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d'}$, with rank $r$ usually only 8–64 and $s$ a scaling factor. With a single adapter, $sAB$ can be merged into $W$ and inference is as fast as the original model; but with several adapters sharing one instance it cannot be merged, since each request needs a different $W$. The approach is to compute $xW$ once for all requests in the base part and compute the LoRA part per request with its own adapter. Two ways to batch it:

- **BGMV** (batched gather matrix-vector): each token "gathers" its adapter's $A$ and $B$ by index and does a small matrix × vector per token. In decode each request has a single token, which fits perfectly;
- **SGMV** (segmented gather matrix-vector): tokens using the same adapter are placed together, and each segment does one small matrix multiply. In prefill a request has many tokens, so a segment is a proper matrix multiply and far more efficient.

![Figure: multi-LoRA serving: the base matrix multiply runs for everyone, while the small LoRA matrices are computed per adapter group and added back](../assets/figures/multi-lora-batch.svg){.aig-svg}

```python
import torch

torch.manual_seed(0)
D_IN, D_OUT, R, N_ADAPTERS = 256, 256, 16, 8
W = torch.randn(D_IN, D_OUT) / 16                                   # the shared base weights
A = torch.randn(N_ADAPTERS, D_IN, R) / 16                           # a pair of low-rank matrices per adapter
B = torch.randn(N_ADAPTERS, R, D_OUT) / 4
SCALE = 2.0                                                         # alpha / r

# a mixed batch: 6 requests with different adapters and different token counts (1 token for decode requests, many for prefill ones)
lengths, adapters = [1, 1, 37, 1, 12, 1], [3, 5, 3, 0, 7, 5]
x = torch.randn(sum(lengths), D_IN)
tok_adapter = torch.tensor([a for a, n in zip(adapters, lengths) for _ in range(n)])


def per_request():
    out, s = [], 0
    for a, n in zip(adapters, lengths):                             # naive: one matrix multiply per request
        xi = x[s:s + n]
        out.append(xi @ W + SCALE * (xi @ A[a]) @ B[a])
        s += n
    return torch.cat(out)


def bgmv():
    """BGMV（batched gather matrix-vector）：每个 token 按自己的适配器编号取 A、B，适合 decode"""
    shrink = torch.einsum("ti,tir->tr", x, A[tok_adapter])          # shrink to rank r first
    return x @ W + SCALE * torch.einsum("tr,tro->to", shrink, B[tok_adapter])


def sgmv():
    """SGMV（segmented gather matrix-vector）：把使用同一个适配器的 token 排在一起，每段做一次小矩阵乘，适合 prefill"""
    order = torch.argsort(tok_adapter, stable=True)
    xs, out = x[order], torch.empty(len(x), D_OUT)
    ids, counts = torch.unique_consecutive(tok_adapter[order], return_counts=True)
    s = 0
    for a, n in zip(ids.tolist(), counts.tolist()):
        out[s:s + n] = SCALE * (xs[s:s + n] @ A[a]) @ B[a]
        s += n
    lora = torch.empty_like(out)
    lora[order] = out
    return x @ W + lora


ref = per_request()
print("BGMV 与逐请求一致：", torch.allclose(bgmv(), ref, atol=1e-5), "；SGMV 与逐请求一致：", torch.allclose(sgmv(), ref, atol=1e-5))
print(f"基座矩阵乘 1 次（{len(x)} 行）；逐请求要调用 {len(lengths)} 次；SGMV 按适配器分成 {len(set(adapters))} 段")
```

```text title="output"
BGMV 与逐请求一致： True ；SGMV 与逐请求一致： True
基座矩阵乘 1 次（53 行）；逐请求要调用 6 次；SGMV 按适配器分成 4 段
```

Real kernels (Punica's BGMV / SGMV and the variants that followed) fuse "gathering the adapter + the small matrix multiply" into one kernel: first shrink ($x \to xA$, down to $r$ dimensions), then expand ($\to (xA)B$) and add directly to the base output. The compute is tiny: a rank-16 LoRA is only about 0.5% of the base compute ([the system design reference answers](../career/design-answers-1.md#4-多租户-lora-服务) work it out), but in decode it is memory-bound: the more distinct adapters a batch uses, the more LoRA weights must be read.

## Scheduling the adapters {#适配器的调度}

| Tier | What it holds | Capacity (8B base, rank 16, all 7 projections) |
| --- | --- | --- |
| GPU memory | the adapters the current batch uses and the most popular ones | about 80 MB each, so 8 GB keeps about 100 resident |
| Host memory | all active adapters | thousands, with a swap-in taking a few milliseconds (PCIe) |
| Local disk / object storage | every adapter | the first load downloads it, hundreds of milliseconds to seconds |

- **Memory management**: S-LoRA puts adapter weights and the KV Cache in one paged memory pool, swapping in and out on demand, avoiding the waste of reserving separately for each;
- **A cap on adapters per batch**: the number of adapters present in one batch is capped (by the slots in GPU memory), and requests beyond it wait for the next batch. This is also a scheduling constraint: when picking requests, the scheduler must consider "is this request's adapter already in GPU memory";
- **Routing affinity**: concentrate requests for the same adapter on a few instances, so each instance's batch has fewer distinct adapters and fewer swaps; replicate popular adapters across several instances;
- **Prefix caching must distinguish adapters**: when LoRA applies to the Q, K and V projections, the same prefix has different KV under different adapters, so a block's hash key must include the adapter's id ([KV transfer and storage](../comm/kv-storage.md)).

In inference frameworks: vLLM turns it on with `--enable-lora`, where `--max-loras` is the maximum adapters per batch, `--max-lora-rank` the maximum rank supported and `--max-cpu-loras` the number cached on the host, with the batched computation in `vllm/lora/punica_wrapper/` and `vllm/lora/ops/`; SGLang uses `--enable-lora`, `--lora-paths`, `--max-loras-per-batch` and `--max-lora-rank`, with `--lora-backend` selecting the kernel (`csgmv` by default) and the implementation in `srt/lora/` (including the memory pool `mem_pool.py` and the eviction policy `eviction_policy.py`).

!!! interview "How to explain it"
    The core of multi-LoRA is "the base computed together, LoRA computed per request": decode uses BGMV (gathering adapters per token), prefill uses SGMV (segmented matrix multiplies by adapter), and real kernels fuse gathering the weights with shrink / expand. Then scheduling: adapters are tens of MB, GPU memory keeps the popular ones resident while the host caches them all, and adapters per batch are capped; routing is affine by adapter; prefix cache keys must include the adapter id. Working out that "LoRA compute is only 0.5%, but the more adapters in a batch, the more weights decode must read" shows you understand where the bottleneck is.

## Exercises {#练习}

**1. When should LoRA be merged into the base weights?**

??? success "Answer"
    When one adapter's traffic is large enough to occupy one or more instances on its own: merged, inference is as fast as the original model, with no LoRA overhead and no adapter slots taken in the batch. The cost is that these instances can serve only that one version, losing the flexibility of mixed batching. A common practice is "merge and deploy the head adapters separately, and serve the long tail with multi-LoRA on shared instances".

**2. What goes wrong when rank-64 and rank-16 adapters are mixed in one batch?**

??? success "Approach"
    Kernels usually allocate and compute by "the maximum rank", so rank-16 adapters are padded to 64, wasting compute and memory; each adapter's footprint in GPU memory is also reserved at the maximum rank. The remedies: schedule by rank group (keeping adapters of similar rank in one batch), or use kernels that support variable ranks; don't set `--max-lora-rank` larger than actually needed.

## Summary {#小结}

- [x] Multi-LoRA doesn't merge weights: the base part is computed together and the LoRA part per request; BGMV suits decode, SGMV suits prefill, and both match per-request computation.
- [x] LoRA's compute is tiny, and the bottleneck is reading different adapters' weights in decode; adapters per batch are capped.
- [x] Adapters live in tiers (GPU memory, host memory, storage), routing is affine by adapter, and prefix cache keys must include the adapter id.
