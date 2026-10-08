# The engine and the sampler

<p class="lead"><code>Engine</code> is the compute engine on one TP rank: it initializes the model, the KV pool, the page table, the attention backend, the sampler and CUDA Graph in a fixed order, and then exposes exactly one method, <code>forward_batch</code>, which takes a prepared batch and returns each request's next token. This chapter finishes it and drives it by hand through prefill and decode with no scheduler at all, matching Hugging Face token by token. That completes the "computing it right" part.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How does the engine decide how many pages the KV pool gets? Why does it measure free memory **before** loading the model?
    2. Why does `forward_batch` return the tokens both on the GPU and on the CPU?
    3. A batch holds some greedy requests and some with `temperature=0.7, top_p=0.9`. How does the sampler handle them in one call?
    4. When top-k and top-p are both set, which comes first?

??? success "Answers (try it yourself first, then expand)"
    1. Pages = (`memory_ratio` × free memory before loading the model − what the model takes) ÷ bytes per page, with the minimum taken across ranks under TP. The early measurement is what reveals how much the model itself takes: the difference between free memory before and after loading.
    2. The GPU copy is written straight back into the token pool as the next step's input, never touching the CPU; the CPU copy (an async transfer) lets the scheduler decide whether a request is finished and hand it to detokenization.
    3. The per-request parameters are laid out as tensors and handled in one batch: when everything is greedy it goes straight to argmax, and in a mixed batch the greedy requests get a tiny temperature (equivalent to argmax) and travel the sampling path with the others.
    4. top-k first, then top-p.

**Files you will write**: `engine/config.py`, `engine/engine.py`, `engine/sample.py`, `utils/device.py` (the device abstraction).

## Configuration {#配置}

@@code python/minisgl/engine/config.py:EngineConfig@@

Most fields map to command-line arguments. `device` and `cpu_kv_cache_bytes` are ours; `model_config` is a `cached_property` parsed from `config.json` on first access (chapter 3).

## The order of initialization {#初始化的顺序}

@@code python/minisgl/engine/engine.py:Engine.__init__@@

The order is deliberate:

1. **TP info and the device.** Every layer built afterwards reads the TP info to decide how it shards.
2. **The stream and the global Context.** On CUDA all of the engine's compute runs on its own stream (overlap scheduling in chapter 11 needs one stream for the scheduler and one for the engine).
3. **Communication.** With TP>1 the process group is created (chapter 16).
4. **Measure free memory, then load the model.** The difference before and after is what the model takes.
5. **The KV pool.** The remaining memory gives the page count (next section), plus one extra page for the dummy request.
6. **The page table**, whose last row belongs to the dummy request and points at the dummy page (`fill_(num_tokens)`, the dummy page's first location).
7. **The attention backend and the MoE backend.** The attention backend reads the KV pool as it is constructed (`get_global_ctx().kv_cache`), so it has to come after the pool.
8. **The sampler and CUDA Graph.** Capturing a graph really runs the model, so it has to be last.

## Planning the memory {#显存规划}

@@code python/minisgl/engine/engine.py:Engine._determine_num_pages@@

Bytes per page is `2 (K and V) × head_dim × this rank's KV heads × page_size × dtype bytes × layers`. Usable memory works out as:

<!-- i18n:diagram 754b127677 -->
```text
usable = memory_ratio × free memory before loading the model − what the model takes
what the model takes = free before loading − free after loading
```

`memory_ratio` defaults to 0.9, leaving 10% for activations, CUDA Graph and various workspaces. Under tensor parallelism the minimum across ranks is taken (`_sync_get_memory` does one all-reduce) so that every rank's KV pool is the same size, because the scheduler makes identical decisions on every rank only if the resources are identical.

On a CPU "free memory" is the whole machine's, and working proportionally would give a KV pool of tens of GB, so we add a cap, `cpu_kv_cache_bytes` (2 GiB by default); tests usually set the page count outright with `num_page_override`.

## forward_batch {#forward_batch}

@@code python/minisgl/engine/engine.py:Engine.forward_batch@@

- run the model inside the `forward_batch` context (or replay the CUDA Graph) to get `[batch_size, vocab]` logits;
- **advance every request's state** with `complete_one()`. Note that this happens before sampling and completes immediately on the CPU, so from here on the scheduler can treat those requests as having one more token and schedule the next step without waiting for the GPU (chapter 11);
- sample to get `next_tokens_gpu`, then copy it to the CPU `non_blocking` and record an event. A caller that needs the CPU values first calls `copy_done_event.synchronize()`.

The two copies serve different purposes: the GPU one is written back into the token pool by the scheduler as the next step's input and never leaves the GPU; the CPU one decides whether a request is finished and goes to the detokenizer.

## The device abstraction {#设备抽象}

Upstream uses `torch.cuda.Stream()`, `torch.cuda.Event()` and `pin_memory=True` directly here. We replace them with a few small functions:

@@code python/minisgl/utils/device.py:create_stream@@

@@code python/minisgl/utils/device.py:NullEvent@@

On a CPU every computation finishes synchronously: by the time `record` runs the result is long since computed, so `synchronize` has nothing to wait for. Pinned memory only means something on CUDA, where it lets host-to-device copies run asynchronously, so `pin(device)` returns `False` on a CPU.

## The sampler {#采样器}

@@code python/minisgl/engine/sample.py:Sampler@@

Sampling takes two steps:

- `prepare(batch)` is called on the scheduler side, **before** the forward pass, and lays each request's sampling parameters out as tensors copied asynchronously to the device. When the whole batch is greedy it returns `temperatures=None` and `sample` goes straight to `argmax`, skipping the softmax.
- `sample(logits, args)` is called after the forward pass.

Requests in one batch may each have their own parameters. The greedy requests in a mixed batch are given a tiny temperature (1e-6), which after the softmax collapses to one-hot, so the sample is the argmax and the whole batch travels one code path. A top-k of -1 (no limit) becomes the vocabulary size, and the tensors for top-k (or top-p) are only built when at least one request in the batch actually sets them.

The PyTorch reference implementation:

@@code python/minisgl/engine/sample.py:sample_torch@@

Like FlashInfer's default, top-k comes first and top-p applies to what is left. top-p keeps the tokens whose preceding probabilities sum to less than p, so the most likely token is always kept. On a GPU it switches to FlashInfer's sampling kernel (`sample_flashinfer`), which uses rejection sampling to avoid a sort and gives the same distribution.

## Driving the engine by hand {#手工驱动引擎}

With no scheduler we play the scheduler ourselves: "allocate" a stretch of KV locations for each request by writing the page table directly, fill in the batch's four fields each step, call `forward_batch`, and append the new token to the request.

@@code examples/ch06_engine.py@@

@@output ch06_engine@@

The first step is prefill: two requests' 9 tokens are packed into one dimension and fed to the model; after that each step decodes 1 token per request. All 8 tokens match Hugging Face's `generate` exactly. What the scheduler of the next part has to do is automate the "allocate KV locations", "fill the batch fields" and "append the token" of this hand-written code, and let requests join and leave at any time.

!!! upstream "The official implementation"
    - initialization: @@upstream engine/engine.py:Engine.__init__@@
    - memory planning: @@upstream engine/engine.py:Engine._determine_num_pages@@
    - the forward pass: @@upstream engine/engine.py:Engine.forward_batch@@
    - sampling: @@upstream engine/sample.py:Sampler@@ (FlashInfer only)

!!! diff "Differences from upstream"
    - Device: upstream supports only CUDA; we support CPU as well through `utils/device.py`.
    - Communication: upstream initializes a gloo process group even at TP=1; we only do so when TP>1, which lets tests create engines repeatedly in one process (`shutdown` clears the global state).
    - CPU thread count: on a CPU we set PyTorch's thread count to the physical core count divided by the TP size, or decode's small matmuls get dragged down tens of times by thread synchronization (see the [architecture chapter](../overview/architecture.md)).

## Tests {#测试}

`tests/test_ch06_engine.py` has three tests: the prefill logits differ from HF by less than 1e-4 at the maximum; the hand-written loop's greedy output matches HF token by token; and the sampler's top-k, top-p and mixed-greedy behaviour and probabilities (with top-k=2 the top token is drawn about e⁵ / (e⁵ + e⁴) ≈ 0.731 of the time).

@@code tests/test_ch06_engine.py:test_sampler_greedy_and_filters@@

!!! interview "How to explain it"
    On the engine: the initialization order is TP info, then the stream and the context, then communication, then loading the model, then the KV pool, the page table, the attention backend, the sampler and CUDA Graph. The KV pool's size is the memory ratio times the free memory before loading, minus what the model takes, converted into pages, with the minimum across ranks under TP (and the measurement has to happen before loading to know what the model takes). `forward_batch` runs the forward pass, advances the request state and samples, returning the tokens on the GPU (written straight back into the token pool) and an asynchronous copy on the CPU (for the scheduler to decide what has finished). The sampler handles requests with different parameters in one call: argmax when everything is greedy, a tiny temperature for the greedy ones in a mixed batch, and top-k before top-p.

## Exercises {#练习}

1. Running Qwen3-0.6B (bf16) on a 24 GB card, with the model taking about 1.2 GB and `memory_ratio=0.9`, roughly how many tokens does the KV pool hold?
2. Add `min_p` to the sampler: keep the tokens whose probability is at least "the maximum probability × min_p". Should it come before or after top-k and top-p?
3. Does it affect correctness if `forward_batch` calls `complete_one()` **after** sampling instead? What about overlap scheduling in chapter 11?

??? success "Answers"
    1. About 0.9 × 24 − 1.2 ≈ 20.4 GB usable, at 112 KiB per token, so roughly 180,000 tokens (CUDA Graph and friends come off the top in practice).
    2. min_p's threshold depends on the maximum probability and usually applies to the full distribution after temperature scaling: compute its mask before top-k and top-p, then intersect it with theirs.
    3. Correctness is unaffected, since `complete_one` only changes counters on the CPU. Overlap scheduling is not fundamentally affected either, because the scheduler assembles the next step only after `forward_batch` returns; what matters is that it does not depend on the sampled values, so the CPU never waits for the GPU.

## Summary {#小结}

- [x] `Engine` initializes in this order: TP info, stream and context, communication, model loading, KV pool, page table, backends, sampler, CUDA Graph.
- [x] The KV pool's size is `memory_ratio × free memory before loading − what the model takes`, converted to pages by the bytes per page, with the minimum taken across ranks under TP.
- [x] `forward_batch` runs the forward pass, advances the request state and samples, returning the tokens on the GPU and an in-flight asynchronous copy on the CPU.
- [x] The sampler supports per-request parameters: argmax when everything is greedy, a tiny temperature for greedy requests in a mixed batch, and top-k before top-p.
