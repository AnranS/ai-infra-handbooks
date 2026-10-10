# Step 0: a minimal inference engine that runs

<p class="lead">Before writing any real module, get an actual inference engine running in under two hundred lines that import nothing from <code>minisgl</code>: read the safetensors, write Qwen3's forward pass by hand, keep the KV cache the most naive way possible, greedy-decode one prompt, check it against Hugging Face token by token, and measure tokens per second. That is the whole skeleton of an inference engine. None of the twenty-odd chapters that follow starts "from zero" — each one adds to this code: swap out one naive part, measure again, and see how much faster it got or how many more requests it can hold.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which two kinds of forward pass make up one "generation"? What are their input lengths?
    2. What does the KV cache store? What happens without it?
    3. In which two ways does Qwen3's attention differ from standard multi-head attention?
    4. The same code goes from 0.4 tokens/s to 27 tokens/s on decode by changing one line. Which line, and why?
    5. Of everything this minimal engine lacks, which one breaks first in a real service?

??? success "Answers (try first, then expand to compare)"
    1. One prefill: the whole prompt in one pass, with input length equal to the prompt's token count; then one decode per generated token, with input length 1.
    2. The K and V computed at every layer (for this book's model: per token, per layer, 8 heads × 128 dims × 2, about 224 KB in fp32). Without it, every step recomputes the whole sequence; step $n$ costs in proportion to $n$, and the total is quadratic.
    3. GQA (16 query heads share 8 KV heads; `repeat_interleave` copies K and V to the matching query heads) and q_norm / k_norm (q and k each go through an RMSNorm before the rotation).
    4. `torch.set_num_threads(physical cores)`. During decode the matrix multiplies have a "batch" of 1 row, and the synchronisation among 32 threads costs far more than the computation; halving the thread count made decode 60× faster.
    5. It serves one request at a time. A real service has dozens to hundreds in flight, and handling them one by one leaves the compute units almost entirely idle — which is what stages 1 and 2 fix.

**Files you will write**: `examples/ch00_tiny_engine.py`, one file, under two hundred lines. It does not go into the `minisgl` package; it is the starting point and the reference for the whole book.

## Run it first {#先跑起来}

```bash
cd minisgl && python examples/ch00_tiny_engine.py
```

On this book's Linux development box the first run takes about a minute — which is itself the first phenomenon this chapter explains (it does not happen on a Mac; see below). The output:

@@output ch00_tiny_engine@@

Three things happened in those few lines: the model really loaded (596M parameters), the generated text is right (`' Paris. The capital of Italy is Rome. ...'`, **identical token by token** to Hugging Face's `generate` over 16 tokens), and we have a first set of numbers — 0.4 tokens/s with the defaults, 26.8 tokens/s after changing one line.

## What the code does {#这份代码在做什么}

@@code examples/ch00_tiny_engine.py@@

The file has four parts, matching the four things an inference engine can never avoid.

**1. Weights.** `config.json` gives the shapes (28 layers, hidden 1024, 16 query heads, 8 KV heads, head_dim 128); `model.safetensors` gives the values. We read every tensor into one `dict` and convert bf16 to fp32 — not because fp32 is better, but to match Hugging Face's fp32 results **bit for bit**, which is the correctness baseline for every chapter in this book. When `tie_word_embeddings` is set, the output layer simply reuses the embedding.

**2. Four operators.** `rmsnorm`, `rope`, `attention` and `mlp`, thirty lines in total:

- RMSNorm has no mean and no bias, only a scale;
- RoPE rotates "the two halves paired" (the Hugging Face and Qwen convention; Meta's original LLaMA pairs adjacent dimensions — see the [LLM Internals handbook](llm://transformer/position/));
- the attention has Qwen3's two peculiarities: **GQA** (`repeat_interleave` turns the 8 KV heads into 16, one per query head) and **q_norm / k_norm** (q and k each go through an RMSNorm before the rotation, which keeps the attention logits from growing too large). The causal mask is computed from absolute positions: the $t$-th query of this step sits at position $S - T + t$ and may only see keys at positions up to its own;
- the MLP is SwiGLU: `down(silu(gate(x)) * up(x))`.

**3. One layer, and the whole model.** `layer` is two rounds of "normalise → sublayer → add the residual". The KV cache is a list of length 28; each entry holds this layer's `(k, v)` for every token so far, and after each step the new ones are appended with `torch.cat`. `forward` returns the logits of **the last position only**: that is all generation needs.

**4. Generation.** `generate` does one prefill (the whole prompt at once, 5 tokens), then 16 decodes (one new token each, positions continuing where the prompt left off). Sampling is the simplest possible greedy `argmax`. The two timers measure prefill and decode separately — these two numbers recur through the whole book, because their bottlenecks are completely different: prefill is matrix-multiply bound, decode is memory bound.

## The first optimisation: measure, then change {#第一个优化先量再改}

With the defaults, decode runs at 0.4 tokens/s, 2.3 seconds per step. The model has only 0.6B parameters and one decode step is roughly 1.2 GFLOP on a CPU; it should be nowhere near that slow. The problem is threads: PyTorch defaults to one thread per *logical* core (32 on this machine), but during decode each matrix multiply has a "batch" of a single row — too little work to split 32 ways, so the synchronisation between threads dominates. After `torch.set_num_threads(16)` decode takes 0.60 seconds, **60× faster**, and not a single output token changed.

This belongs in step 0 because it demonstrates the book's entire attitude towards "optimisation": **have a measurable baseline, change one thing, measure again, and the output must not change.** The tests and benchmarks at the end of every later chapter are that sentence spelled out. Incidentally, the real `Engine` (chapter 6) sets the thread count to "physical cores / TP size" automatically on a CPU, so you will not have to do this by hand again.

!!! tip "On your own machine: the same code measured twice"
    | Machine | Default threads | Decode with defaults | `cpu_count() // 2` | Decode after |
    | --- | --- | --- | --- | --- |
    | Linux dev box, 32 logical cores (hyper-threaded) | 32 | 0.4 tokens/s | 16 | 26.8 tokens/s |
    | Apple Silicon Mac, 12 cores (performance + efficiency) | 8 | 37.6 tokens/s | 6 | 38.9 tokens/s |

    There is **no trap on a Mac**: on macOS PyTorch defaults to the performance cores only (8 on this machine) rather than every efficiency core and hyper-thread, so the very first run does 37 tokens/s and 6 threads merely match it. The trap exists only on platforms where "default threads = all logical cores", which is exactly what Linux servers are. Note too that this Mac decodes faster than the 32-core Xeon: decode is memory bound, every step reads the 2.4 GB of fp32 weights once, and Apple's unified memory has more bandwidth than server DDR — whoever has the bandwidth wins, a conclusion that will keep coming back once we reach GPUs.

    The thread-count line's `os.cpu_count() // 2` assumes two logical cores per physical core; Apple Silicon has no hyper-threading, and halving merely happens to land near the performance-core count. The real method is to try both and keep the faster one — which is "measure, then change" again.

## What it lacks: the map of the whole book {#它缺什么整本书的路线图}

This engine produces correct tokens, but it is a long way from an inference *system*. Each row below is one naive choice in this code, and the chapter where this book replaces it. Once you have read this table, the structure of the whole book is clear.

| What this code does | The problem | What replaces it | Where |
| --- | --- | --- | --- |
| KV rebuilt with `torch.cat` every step | step $n$ copies $n$ tokens' worth of KV, so total traffic is quadratic; at 224 KB per token in fp32, 1024 tokens is 224 MB, moved on every step | a pre-allocated paged KV pool with a page table; a new token writes only its own page | Stage 1 [The KV pool](../compute/kvcache.md) |
| one request at a time | the compute units sit almost idle: a decode step's matrix multiply has a single row | `Req` / `Batch`: tokens from many requests assembled into one batch | Stage 1 [Core data structures](../compute/core.md), Stage 2 [The scheduler](../schedule/scheduler.md) |
| all weights read into a `dict` at once, looked up by name | large models do not fit in memory; each layer's weight names are scattered everywhere | streaming loading plus an op layer where each `Layer` owns its weights and their loading | Stage 1 [The op layer](../compute/layers.md), [Models and weights](../compute/models.md) |
| attention computes the full $T \times S$ score matrix | prefilling 4096 tokens, the score matrices of 16 heads take 1 GiB; the mask is materialised too | pluggable attention backends: the CPU reference, FlashInfer, FlashAttention | Stage 1 [Attention backends](../compute/attention.md), Stage 4 [GPU attention](../perf/gpu-attention.md) |
| sampling is just `argmax` | no temperature, top-k or top-p, and no per-request settings | a batched sampler | Stage 1 [The Engine and the sampler](../compute/engine.md) |
| a prefix shared by two requests is computed twice | system prompts and multi-turn history are recomputed every time | the radix cache: reuse KV by prefix | Stage 2 [Radix Cache](../schedule/radix-cache.md) |
| a long prompt is prefilled in one go | one 8K prefill stalls every decoding request for seconds | chunked prefill | Stage 2 [Chunked prefill](../schedule/chunked-prefill.md) |
| Python scheduling and compute run serially | every step waits for the CPU to prepare the next | overlap scheduling (CPU side), CUDA Graph (launch side) | Stage 2 [Overlap scheduling](../schedule/overlap.md), Stage 4 [CUDA Graph](../perf/cuda-graph.md) |
| one script, one process | no HTTP, no streaming, no tokenizer process | ZMQ messages, tokenizer / detokenizer processes, an OpenAI-compatible API | Stage 3 [Serving](../serve/message.md) |
| a single GPU only | the model does not fit in one card's memory | tensor parallelism: four ways to shard a linear layer | Stage 4 [Tensor parallelism](../perf/tensor-parallel.md) |

At the end of every stage you hold an engine that is stronger than the last one, still runs, and still matches Hugging Face token by token.

!!! interview "How to explain it"
    To answer "what is the core of an inference engine", this file is enough: four things — weight loading, the forward pass (normalisation, attention, MLP, residuals), the KV cache and the sampling loop — and two kinds of forward pass, prefill and decode. Then explain why it cannot go straight into production: one request at a time, KV copied every step, the attention score matrix materialised, no prefix reuse — and each of those maps to a module in a mature engine (batching, paged KV, FlashAttention, prefix caching). Being able to walk the line "naive choice → problem → corresponding module" is what shows you really understand why inference systems look the way they do.

## Exercises {#练习}

1. Remove the KV cache from `generate` (re-`forward` all tokens at every step), time the 16 decode steps, and compare with the cached version. How big is the gap with a 200-token prompt?
2. Print the shape and byte size of `scores` inside `attention`, once during prefill and once during decode. How large is the prefill score matrix for a 4096-token prompt?
3. Sweep the argument of `torch.set_num_threads` from 1 to the logical core count and plot decode tokens/s. Where is the knee, and how does it relate to the physical core count?

??? success "Answers"
    1. Without the cache, step $n$ computes $5 + n$ tokens, so 16 steps run forward passes over about 200 tokens in total, versus 16 with the cache; with a 200-token prompt the gap is more than tenfold. That is the entire reason the KV cache exists — spend memory to remove quadratic compute.
    2. During prefill `scores` is `[16, T, T]`; at T = 4096 that is $16 \times 4096^2 \times 4$ bytes = 1 GiB, and that is one layer of one batch. During decode it is `[16, 1, S]`, tiny. The whole point of FlashAttention is never to materialise the former.
    3. The curve rises then falls, with the knee usually near the physical core count; beyond it, synchronisation costs more than the parallelism earns. The same logic decides how many cores each instance gets in a multi-GPU or multi-instance deployment.

## Summary {#小结}

- [x] The skeleton of an inference engine is four things — weights, forward pass, KV cache, sampling loop — and two forward passes, prefill and decode.
- [x] Qwen3's attention has two peculiarities, GQA and q_norm / k_norm; RoPE pairs the two halves.
- [x] The optimisation method: a measurable baseline, one change, measure again, output unchanged. One thread-count line made decode 60× faster.
- [x] Every naive choice in this file maps to a later chapter; at the end of every stage you hold an engine that runs.
