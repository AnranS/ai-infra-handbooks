# A walkthrough of SGLang

<p class="lead">SGLang solves the same problem as vLLM, with a similar structure: a frontend process handles HTTP and tokenization, scheduler processes handle scheduling and execution, and there is a separate detokenization process. Its distinctive features are the radix tree prefix cache, a scheduling loop that overlaps CPU and GPU, and a complete set of parallelism and PD-disaggregation designs for large-scale MoE models such as DeepSeek. Based on SGLang 0.5.20, this chapter reads the main line along a request's path and compares it with vLLM.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Which process is each of `TokenizerManager`, `Scheduler` and `DetokenizerManager` in? How do requests and results flow between them?
    2. What is the difference between `ScheduleBatch` and `ForwardBatch`?
    3. How does `event_loop_overlap` overlap CPU scheduling with GPU computation? When is the overlap turned off?
    4. Why does SGLang's KV cache use the two-level structure of `ReqToTokenPool` and `TokenToKVPoolAllocator`?

??? success "Answers (try first, then expand to compare)"
    1. `TokenizerManager` is in the main process (together with the HTTP server), tokenizing and sending requests to the scheduler; `Scheduler` is in the scheduler processes, one per TP rank (which also run the model); `DetokenizerManager` is a process of its own. Requests: main process → scheduler; results: scheduler → detokenizer process → main process, all over ZMQ.
    2. `ScheduleBatch` is the scheduling layer's data structure on the CPU (the list of requests and scheduling metadata); `ForwardBatch` holds the GPU tensors prepared for one forward pass (input_ids, positions, KV locations, attention metadata), converted from the former.
    3. It submits this batch to the GPU first, then processes the previous batch's results on the CPU (detokenization, checking for finished requests, preparing the next batch), so CPU scheduling overlaps GPU computation. When two consecutive batches are both prefill, the overlap is off by default: otherwise the first batch's first token would only be processed and sent after the second batch was submitted, stretching TTFT by a whole batch (controlled by `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`); for debugging, `--disable-overlap-schedule` turns it off entirely.
    4. `ReqToTokenPool` records which KV slot each position of each request uses (request → token position → slot), and `TokenToKVPoolAllocator` manages allocating and freeing slots (slot → the actual KV data). The two levels separate "which slots a request uses" from "slot allocation": sharing slots through the prefix cache, freeing on request completion, and the radix tree deciding what to keep all only change the mapping.

!!! note "Version"
    This chapter is based on the `sglang/srt/` directory of SGLang 0.5.20 (September 2026). SGLang also moves fast and has recently split quite a lot of logic into `managers/scheduler_components/`, subdirectories of `mem_cache/` and `arg_groups/`, while class names and the main structure are relatively stable.

## Getting ready {#准备工作}

```bash
pip download sglang==0.5.20 --no-deps -d . && python -m zipfile -e sglang-0.5.20-*.whl sglang-src
# or: git clone https://github.com/sgl-project/sglang && git checkout v0.5.20
```

Start a server: `python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 30000`. Common debugging flags are `--disable-cuda-graph`, `--disable-overlap-schedule` and `--log-level debug`; `python -m sglang.benchmark.one_batch` (the old path `sglang.bench_one_batch` is deprecated) runs a forward pass on a single batch directly without starting a server, which is good for stepping through model execution.

## A map of the directories {#目录地图}

| Directory | Contents |
| --- | --- |
| `srt/entrypoints/` | the HTTP server (`http_server.py`), the offline `Engine` (`engine.py`, which launches the subprocesses), the OpenAI-compatible API (`openai/`) |
| `srt/managers/` | `TokenizerManager`, `Scheduler`, `DetokenizerManager`, `TpModelWorker`, batch data structures (`schedule_batch.py`), scheduling policies (`schedule_policy.py`), `scheduler_components/` (result processing, output streaming and other components split out of the scheduler) |
| `srt/mem_cache/` | KV memory pools and allocators, the radix tree prefix cache, hierarchical caching (HiCache) |
| `srt/model_executor/` | `ModelRunner`, `ForwardBatch`, CUDA Graph runners |
| `srt/layers/` | parallel linear layers, `RadixAttention`, attention backends (`attention/`), MoE (`moe/`), quantization, the sampler |
| `srt/models/` | implementations of each model |
| `srt/speculative/` | speculative decoding: EAGLE, MTP, n-gram and more |
| `srt/disaggregation/` | PD disaggregation: prefill/decode-side logic and transfer backends such as Mooncake and NIXL |
| `srt/constrained/` | structured output (grammar backends such as xgrammar) |
| `srt/eplb/`, `srt/elastic_ep/` | expert load balancing, elastic expert parallelism |

High-performance kernels are in the separate `sgl-kernel` package; routing across instances is handled by SGLang Model Gateway (formerly sgl-router), written in Rust.

## Process structure {#进程结构}

![Figure: SGLang's process structure: TokenizerManager, Scheduler, DetokenizerManager](../assets/figures/sglang-processes.svg){.aig-svg}

<!-- i18n:diagram 2bdb0a56fb -->
```text
Main process                             Scheduler processes (one per TP rank)       DetokenizerManager process
┌────────────────────────────┐    ZMQ    ┌───────────────────────────────┐    ZMQ    ┌────────────────────────────┐
│ HTTP server (FastAPI)      │ ────────▶ │ event_loop_overlap / normal   │ ────────▶ │ incremental detokenization │
│ TokenizerManager           │           │  ingest → batch → run_batch   │           │ stop strings               │
│  tokenize, chat template,  │           │  TpModelWorker → ModelRunner  │           │                            │
│  multimodal preprocessing  │           │                               │           │                            │
└──────────────▲─────────────┘           └───────────────────────────────┘           └──────────────┬─────────────┘
               │                      results back to TokenizerManager (ZMQ)                        │
               └────────────────────────────────────────────────────────────────────────────────────┘
```

The biggest structural difference from vLLM: SGLang's scheduler **runs the model in the same process** (each TP rank has its own Scheduler process, and each schedules the same batch), with no separate "EngineCore → worker" layer; detokenization, on the other hand, has a dedicated process. Under data parallelism (`--dp-size`), a `DataParallelController` in front dispatches requests to several schedulers.

## Main line 1: TokenizerManager {#主线一tokenizermanager}

`TokenizerManager.generate_request()` (`srt/managers/tokenizer_manager.py`) does much what vLLM's frontend does: `_tokenize_one_request` tokenizes (and preprocesses multimodal input), builds the request object and sends it to the scheduler over ZMQ, then waits for the request's results. Results are received from DetokenizerManager by `handle_loop`, `_handle_batch_output` dispatches them to each request's waiter, and the HTTP layer turns them into streaming responses.

## Main line 2: the Scheduler's event loop {#主线二scheduler-的事件循环}

The scheduler process is started by `run_scheduler_process` and enters one of two event loops (`srt/managers/scheduler.py`):

```py
def event_loop_normal(self):                     # normal loop: ingest → form a batch → run → process results
    while True:
        self.ingest_requests()
        plan = self.get_next_batch_to_run(running_batch=self.running_batch, last_batch=self.last_batch)
        batch = plan.batch_to_run
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        self.last_batch = batch
```

```py
def event_loop_overlap(self):                    # overlap loop (default): submit this batch first, then process the previous batch's results
    while True:
        self.ingest_requests()
        plan = self.get_next_batch_to_run(...)
        batch = plan.batch_to_run
        if batch:
            batch_result = self.run_batch(batch)                 # submitted to the GPU asynchronously; returns immediately
            self.result_queue.append((batch.copy(), batch_result))
        if self.last_batch:
            pop_and_process()                                    # process the previous batch's results while the GPU computes this one
        self.launch_batch_sample_if_needed(batch_result, batch)
        self.last_batch = batch
```

The key to the overlap loop: **while the GPU runs batch N, the CPU processes batch N−1's results and prepares batch N+1**. Batch N's input depends on the tokens just sampled for batch N−1, which the CPU does not know yet; SGLang solves this with placeholders for "future tokens" (the results are written directly on the GPU into the next batch's input buffer). The price is that results are processed one step later, so the overlap is turned off for a step between two consecutive prefill batches, and when structured output needs to synchronize grammar state (`is_disable_overlap_for_batch`). vLLM's asynchronous scheduling is the same idea.

`get_next_batch_to_run()` decides in this order:

1. Merge the requests that just finished prefill in the previous step into the running batch (`running_batch`);
2. Try to form a new prefill batch (`get_new_batch_prefill`): sort the waiting queue by the scheduling policy (`SchedulePolicy`, LPM by default), and use `PrefillAdder` to admit requests one by one within the remaining token budget and memory headroom, chunking very long requests (`chunked_prefill_size`);
3. If no prefill batch can be formed, let the running batch do one decode step (`update_running_batch`); if memory runs short, `retract_decode` withdraws some requests.

In other words, SGLang is **prefill-first** by default: if new requests can be prefilled, prefill goes first, and decode requests pause for that step; only with `--enable-mixed-chunk` are prefill chunks and decode merged into the same batch (`ForwardMode.MIXED`). This differs from vLLM's "every request within the same token budget", and is an important difference in the two schedulers' behavior.

`process_batch_result()` handles the forward results: `process_batch_result_prefill` and `process_batch_result_decode` in `scheduler_components/batch_result_processor.py` append tokens to requests, check stop conditions, and insert finished requests' KV into the radix tree (`cache_finished_req`), and `stream_output` in `output_streamer.py` sends the increments to DetokenizerManager.

## Main line 3: batch data structures {#主线三批次数据结构}

The comment at the top of `schedule_batch.py` describes a batch's data flow, which can be summarized as:

> `ScheduleBatch` → `ForwardBatch`. `ScheduleBatch` is managed by the scheduler and holds high-level scheduling data, mostly on the CPU; `ForwardBatch` is managed by `ModelRunner` and holds low-level tensor data, mostly GPU tensors, built directly from a `ScheduleBatch` by `ForwardBatch.init_new`.

- **`Req`**: all of a request's state (input tokens, output tokens, the prefix match result `prefix_indices`, the finish reason…);
- **`ScheduleBatch`**: a batch of `Req`s with their scheduling information; `forward_mode` says whether it is `EXTEND` (prefill), `DECODE`, `MIXED` and so on;
- **`ForwardBatch`** (`srt/model_executor/forward_batch_info.py`): `input_ids`, `positions`, `seq_lens`, `extend_prefix_lens`, `extend_seq_lens`, `req_pool_indices`, `out_cache_loc` (the slot mapping)… corresponding to the mini engine's `BatchInput`.

## Main line 4: KV memory and the prefix cache {#主线四kv-内存与前缀缓存}

SGLang manages KV in two levels:

- **`ReqToTokenPool`** (`mem_cache/memory_pool.py`): a `[max requests, max context length]` table whose row r, column i is the KV slot of request r's token i. It is the equivalent of vLLM's block table, only at the granularity of tokens (or pages) instead of blocks;
- **`TokenToKVPoolAllocator`** (`mem_cache/allocator/`): manages free slots, "give me n slots";
- **`MHATokenToKVPool`** and other `KVCache` subclasses: the actual K/V tensors, read and written by `(layer_id, slot)`.

The radix tree `RadixCache` (`mem_cache/radix_cache.py`) records "which token sequences have their KV in which slots"; see [the prefix caching chapter](../engine/prefix-cache.md). When a request is admitted, `match_prefix` yields `prefix_indices`, and new slots are allocated only for the rest; when it finishes, `cache_finished_req` inserts its KV into the tree instead of freeing it directly. When memory runs short, the allocator first has the radix tree evict leaves no longer in use (`evict`). `mem_cache/README.md` lays out this layer's complete hierarchy (allocation policy → multi-pool routing → allocator → device pool → host pool → storage backend).

## Main line 5: model execution {#主线五模型执行}

`run_batch` calls `TpModelWorker.forward_batch_generation` (`srt/managers/tp_worker.py`), which builds a `ForwardBatch` and hands it to `ModelRunner.forward` (`srt/model_executor/model_runner.py`). `_forward_raw` first checks whether the decode CUDA Graph can be used (`decode_cuda_graph_runner.can_run_graph`) and replays it if so; otherwise it runs an ordinary extend or decode forward pass according to `forward_mode`.

The model code (such as `srt/models/qwen2.py`) is very similar to vLLM's: the same merged QKV, merged gate/up, fused `RMSNorm(x, residual)`, and column- and row-split parallel linear layers (early SGLang reused these layers from vLLM directly). The attention layer is `RadixAttention` (`srt/layers/radix_attention.py`), which does not compute itself but hands `q, k, v` and `forward_batch` to the current attention backend (`flashinfer_backend.py`, `flashattention_backend.py`, `triton_backend.py`, `flashmla_backend.py` and others under `srt/layers/attention/`). The default backend is chosen automatically by hardware and model, the choice is printed in the startup log, and it can be set with `--attention-backend`, or separately with `--prefill-attention-backend` and `--decode-attention-backend`.

## SGLang vs. vLLM {#sglang-与-vllm-对照}

| | vLLM V1 | SGLang |
| --- | --- | --- |
| Processes | frontend / EngineCore / one worker per GPU | main process (frontend) / one Scheduler per TP rank (including model execution) / detokenizer |
| Scheduling | a unified token budget; prefill and decode always in the same step | prefill-first by default; mixed batches optional |
| Memory admission | admit if allocation succeeds, preempt when short | estimate future demand (`new_token_ratio`), admit conservatively, retract when necessary |
| Prefix cache | chained-hash blocks + LRU free queue | radix tree + leaf LRU + reference locks |
| Scheduling policies | FCFS, priority | LPM, DFS-weight (cache-aware), FCFS, LOF and others |
| CPU/GPU overlap | asynchronous scheduling (`AsyncScheduler`), the batch queue | `event_loop_overlap` |
| Batch data | `SchedulerOutput` (increments) → persistent batch `InputBatch` | `ScheduleBatch` → `ForwardBatch` |
| Large-scale MoE | DP + EP, DeepEP, EPLB | DP attention + EP, DeepEP, EPLB, two-batch overlap (TBO) |

The two keep borrowing from each other; this table describes the default behavior as of September 2026, and the source is the final word on details.

## A suggested reading order {#建议的阅读顺序}

1. `_launch_subprocesses` in `srt/entrypoints/engine.py`: see which processes there are and how they are launched;
2. `Scheduler.event_loop_normal` → `get_next_batch_to_run` → `get_new_batch_prefill` (with `PrefillAdder` in `schedule_policy.py`) → `update_running_batch`;
3. `schedule_batch.py`: `Req`, `ScheduleBatch` and the methods that prepare extend/decode;
4. `mem_cache/radix_cache.py` and `memory_pool.py`;
5. `ModelRunner.forward` → `RadixAttention` → one attention backend;
6. Then `event_loop_overlap`, to understand the timing of overlap scheduling.

!!! interview "In an interview"
    "How do SGLang and vLLM differ?" Don't stop at "one uses a radix tree, the other hashes". Pick three points from the comparison table: **process structure** (SGLang's scheduler and model execution are in the same process), **scheduling behavior** (prefill-first vs. a unified budget, conservative vs. aggressive admission with preemption) and **prefix caching** (a radix tree with cache-aware scheduling vs. hashed blocks). Finish with "the two keep borrowing from each other; both overlap CPU and GPU, and both support PD disaggregation and large-scale EP", which shows you have read recent code.

## Exercises {#练习}

**1. The cost of overlap scheduling.** Why does SGLang turn off the overlap between "two consecutive prefill batches" by default?

??? success "Answer"
    Overlap scheduling processes batch N's results one step later. For prefill, processing the results includes sending the first token; if two consecutive batches are both prefill, the first batch's first token is only processed and sent after the second batch is submitted, stretching TTFT by a whole batch. For the sake of TTFT, SGLang finishes processing the previous batch first in this case (controlled by `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`).

**2. The cost of token granularity.** `ReqToTokenPool` records a slot number for every token of every request. Compared with vLLM recording one block number per 16 tokens, what does this cost? How does SGLang mitigate it?

??? success "Approach"
    16 times the metadata: larger block tables, slower to build and copy; more importantly, attention kernels address indirectly per token, and K/V may be completely non-contiguous in memory, so memory access is inefficient. The mitigation is `--page-size` (for example 16, 32 or 64), allocating and matching by page, with radix tree matching aligned to pages as well. Some attention backends (FlashMLA, TensorRT-LLM MLA) require a specific page size in the first place.

## Summary {#小结}

- [x] SGLang's main process handles HTTP and tokenization, each TP rank has one scheduler process (which also runs the model), and detokenization has a process of its own.
- [x] The scheduling loop overlaps CPU and GPU work by default; batch formation is prefill-first, and decode requests are retracted when memory runs short.
- [x] `ScheduleBatch` (CPU, scheduling layer) → `ForwardBatch` (GPU tensors); KV is managed in two levels, `ReqToTokenPool` + the slot allocator, with the radix tree deciding what to keep.
- [x] Compared with vLLM: the process split, scheduling behavior and prefix cache structure differ, while the core ideas (continuous batching, paged/slotted KV, CPU/GPU overlap) are the same.
