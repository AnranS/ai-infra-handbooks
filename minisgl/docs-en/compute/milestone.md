# Stage 1 milestone: the same output, the real structure

<p class="lead">The six chapters of stage 1 replace every naive part of step 0's 126 lines with a real module: <code>Req</code> / <code>Batch</code> / <code>Context</code>, an op layer that does not depend on <code>nn.Module</code>, a model with streaming weight loading, a paged KV pool, pluggable attention backends, and an <code>Engine</code> with a sampler. This page runs the end state first — the same prompt, the same 16 tokens, plus one thing the skeleton could not do: four requests in one batch, computed in one step. Come back to these numbers after the six chapters.</p>

**What you hold at the end of this stage**

- An `Engine`: give it a `Batch` with its fields filled in and it returns each request's next token;
- No scheduler yet — you fill the batch's fields yourself (exactly the lines stage 2 automates);
- A single request is as fast as step 0; several requests computed together give three times step 0's throughput;
- Every step's output still matches Hugging Face token by token.

## Run it first {#先跑起来}

```bash
cd minisgl && python examples/stage1_milestone.py
```

@@code examples/stage1_milestone.py@@

@@output stage1_milestone@@

## Reading the numbers {#读这几个数字}

**A single request: 28.1 tokens/s, versus 26.8 in step 0.** The real structure **gains nothing** on a single request — the difference is run-to-run noise. No surprise: paged KV writes are scattered by position, the attention backend has to slice out each request's own KV, and the `Req` / `Batch` bookkeeping is all Python. What those costs buy is invisible in the single-request number; they exist for the next line.

**Four requests in one batch: 58 ms per step, only 63% slower than the 36 ms of a single request, 68.9 tokens/s in total.** This is what step 0's skeleton **could not do**. Decode is memory bound: one step reads the 2.4 GB of fp32 weights from memory once, and whether the matrix multiply's "batch" has 1 row or 4 makes almost no difference to that read. So gathering requests into a batch adds only the small extra compute per step, while the weights are read once. That is the whole argument for continuous batching, and where an inference system's throughput comes from.

**79 KV pages written, no copies at all.** Step 0 copied the entire KV with `torch.cat` on every step; now each token writes only its own page, and the page table records where it is. Run alone or inside the batch, the first request's 16 tokens are identical — batching changes no request's result, something every later chapter's tests confirm again and again.

## What replaced each part of the skeleton {#骨架的每一块换成了什么}

| Step 0 | Stage 1 | Why |
| --- | --- | --- |
| one `ids` tensor, one `pos` tensor | [`Req`, `Batch`, `Context`](core.md) | requests have their own lengths and state; the model reads the current batch from the global `Context`, so `forward()` takes no arguments |
| four functions plus a weight `dict` | [The op layer: `BaseOP` and the `Layer`s](layers.md) | each `Layer` owns its weights' shapes, loading and sharding — the only way tensor parallelism can split them later |
| one `safe_open` reading everything | [Models and weight loading](models.md) | streaming loading on demand; one decoder file covers Qwen3, Qwen2.5 and Llama 3 |
| `list` + `torch.cat` | [The KV pool, the page table and the token pool](kvcache.md) | pre-allocated, written by page, zero copies; many requests share one pool |
| `einsum` + an explicit mask | [Attention backends and the reference implementation](attention.md) | metadata for variable-length batches, paged KV and prefix reuse; FlashInfer / FlashAttention on a GPU |
| `argmax` | [The Engine and the sampler](engine.md) | temperature, top-k and top-p per request, sampled in one batched call; `Engine` initialises everything above in the right order |

## How to read these six chapters {#怎么读这六章}

In order: read the self-test, write your own version, compare with this book's and the official implementation, then run the chapter's tests. After [The Engine and the sampler](engine.md), come back and rerun this page's script: the few lines in `step()` that "fill the batch fields by hand" each correspond to one duty of stage 2's scheduler — where `positions` and `out_loc` come from (page allocation), why `cached_len` changes (prefix caching), who decides which requests enter this step's batch (admission control). Stage 2 is those lines, automated.

!!! abstract "Checkpoint: do these before stage 2"
    - [ ] Without looking, draw `Req`'s three length fields before and after prefill;
    - [ ] Explain what one row of the page table is and how `out_loc` is computed;
    - [ ] Rerun this page's script with `page_size=4` and get the same output (chapter 4's exercise);
    - [ ] Say why four requests per step cost far less than four times one.

## Summary {#小结}

- [x] Stage 1 replaces the skeleton's six naive parts with six real modules whose interfaces match the official ones.
- [x] A single request is no faster than the skeleton; several requests in one batch are the point of the real structure — decode reads the weights once, however many rows it computes.
- [x] KV is written by page with zero copies; batching changes no request's result.
- [x] The lines filled by hand in `step()` are what stage 2's scheduler does.
