# The gap to SGLang, and extensions

<p class="lead">You now have a complete mini-sglang: an OpenAI-compatible service, a radix cache, chunked prefill, overlap scheduling, tensor parallelism, three attention backends, CUDA Graph and MoE. It shares the official structure and interfaces, and it fixes several problems in the official code. This chapter takes stock of what separates it from production SGLang, with every item a direction you could keep going in, and where in the code to start.</p>

## A recap: the code one request passes through {#回顾一个请求经过的代码}

| Step | Code | Chapter |
| --- | --- | --- |
| the HTTP request arrives and gets a uid | `v1_chat_completions` in `server/api_server.py` | 15 |
| the template is applied and the text tokenized | `tokenizer/tokenize.py` | 13 |
| it joins the waiting queue | `_process_one_msg` in `scheduler/scheduler.py` | 7 |
| prefix matching, admission, request slot | `PrefillAdder` in `scheduler/prefill.py` | 8, 9 |
| chunking | `PrefillAdder._add_one_req` | 10 |
| padding, pages, positions, metadata | `Scheduler._prepare_batch` | 7, 8, 18 |
| the forward pass (or a replay), then sampling | `Engine.forward_batch` | 6, 18 |
| attention | `attention/*.py` | 5, 17 |
| tensor-parallel communication | `layers/linear.py`, `distributed/` | 16 |
| handling results, caching the prefix, releasing | `Scheduler._process_last_data`, `CacheManager.cache_req` | 7, 9, 11 |
| incremental detokenization | `tokenizer/detokenize.py` | 13 |
| streaming back over SSE | `FrontendManager.stream_chat_completions` | 15 |

## The official problems this book fixes {#本书修正的官方问题}

Five problems were found and fixed along the way. Each has a test: put the official version back and the test fails. They could be written up as issues or pull requests for the official repository:

| Problem | Effect | Chapter |
| --- | --- | --- |
| using `can_decode` to judge `max_tokens` under overlap scheduling | the finished flag arrives one token early, so an online service returns one token fewer | [Overlap scheduling](../schedule/overlap.md) |
| the extra step after EOS still sends a reply | the detokenizer creates state for a finished request and never releases it | [Overlap scheduling](../schedule/overlap.md) |
| a finished request's slot is reused at once | the scheduler stream and the engine stream race to write the same token-pool row | [Overlap scheduling](../schedule/overlap.md) |
| an abort while prefill is in flight | `cache_req` runs again on an already released request, and the radix cache frees pages twice | [Overlap scheduling](../schedule/overlap.md) |
| PUB/SUB does not wait for subscribers | the first broadcast can be lost and the other ranks block forever (it only works thanks to startup timing) | [Messages and ZMQ](../serve/message.md) |

Two more are worth mentioning: the official radix cache's integrity check is empty, and we implemented one that really walks the tree (chapter 9); and the official YaRN does not multiply by the attention scaling factor from the Hugging Face implementation, which is why we left YaRN as an exercise rather than implementing it (chapter 2).

## What separates this from production SGLang {#与-sglang-正式版的差距}

**Scheduling**

- **Mixed batches**: mini-sglang does only prefill or only decode per step, so decode pauses while a long prompt is chunked. Production puts decoding requests and the prefill chunk in one batch. Start from `Scheduler._schedule_next_batch` (exercise 1 of chapter 10).
- **Preemption and recomputation**: mini-sglang reserves KV for the worst case and never preempts, at the price of lower concurrency. Production admits optimistically and retracts requests when KV runs short, relying on the radix cache to make recomputation cheap. Start from `PrefillAdder._try_allocate_one` and `_prepare_batch` (exercise 2 of chapter 8).
- **Cache-aware scheduling policies**: production has `lpm` (longest prefix match first), `dfs-weight` and others that favour requests which would hit more cache; mini-sglang is first come first served. Start from the traversal order in `PrefillManager.schedule_next_batch`.

**KV cache**

- **Tiered caching (HiCache)**: move evicted KV to CPU memory or even disk rather than dropping it. The official `MatchResult` leaves a "TODO: support HiCache".
- **MLA and other attention variants**: `create_kvcache_pool` only supports MHA/GQA (the official comment says "TODO: support other variants (e.g. MLA)"). Supporting DeepSeek needs a new KV pool layout and attention backend.
- **KV cache quantization**: an FP8 KV cache doubles the capacity.

**Distributed**

- **Prefill/decode disaggregation**: put prefill and decode on different instances and transfer the KV between them (the idea is in the [Inference Systems handbook](serving://distributed/pd-disagg/)).
- **Expert parallelism and DP attention**: the mainstream way to deploy MoE models; mini-sglang has only tensor parallelism.
- **Routing**: a cache-aware router in front of several instances (SGLang's sgl-router).

**Decoding features**

- **Speculative decoding**: EAGLE, MTP and others, which need a draft model, tree verification and KV rollback.
- **Structured output**: grammar-constrained sampling (xgrammar), which needs a mask over the logits before sampling.
- **More sampling parameters**: stop strings, repetition penalties, logprobs, parallel sampling with `n > 1`, min-p. The current `SamplingParams` has only temperature, top-k, top-p, `ignore_eos` and `max_tokens`.

**Everything else**

- Quantized models (FP8, AWQ, GPTQ), LoRA, multimodality, observability (Prometheus metrics), health checks and graceful shutdown.

## Extensions {#扩展练习}

In order of difficulty, each one usable as a portfolio project (how to write them up is in the [portfolio chapter of the Inference Systems handbook](serving://career/projects/)):

1. **Fill in the sampling parameters**: implement stop strings, `min_p` and `logprobs`. It touches `SamplingParams`, `Sampler`, `DetokenizeMsg` and the API server.
2. **Send a PR upstream**: pick one of the problems this book fixes, write a minimal reproduction in the official repository (on a GPU), and submit the fix. Read the official contributing guide first, and describe the reproduction and the test results in the PR.
3. **Mixed batches**: let the prefill chunk share a batch with the decoding requests, and use chapter 21's load test to compare P99 TPOT on long prompts.
4. **Preemption**: implement optimistic admission plus preemption, and compare concurrency and throughput at the same KV capacity.
5. **HiCache**: copy evicted KV to CPU memory and copy it back on a hit. It needs `RadixTreeNode` to distinguish values on the GPU from values on the CPU.
6. **A prefill/decode disaggregation prototype**: two processes, one doing only prefill and one only decode, passing KV through shared memory (on a CPU) or NCCL (on a GPU).

## Talking about it in an interview {#面试怎么讲}

!!! interview "How to present the engine you built"
    Walk through "the journey of one request": the process layout, the scheduler's four managers, continuous batching and admission control, the radix cache, overlap scheduling, tensor parallelism, CUDA Graph. Give one number per part ("with a shared 400-token prefix the prefill work drops to a tenth", "at TP=2 one forward pass does 57 all-reduces") and have one detail ready to go deep on: the problems under overlap scheduling show best that you really understand the code, so be ready to say why the state runs a step ahead, what that causes, how to fix it and how to test it.

## Summary {#小结}

- [x] Every step a request takes maps to one chapter of this book and one file in the official repository.
- [x] This book fixes five problems in the official code, each with a test that reproduces it.
- [x] The gap to production is concentrated in scheduling policy (mixed batches, preemption, cache awareness), the KV cache (tiering, MLA, quantization), distribution (prefill/decode disaggregation, expert parallelism) and decoding features (speculative decoding, structured output).
