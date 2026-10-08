# A walkthrough of vLLM V1

<p class="lead">vLLM has hundreds of thousands of lines of code, but the main line a request actually travels has only five segments: the frontend (HTTP → AsyncLLM), the EngineCore main loop, the scheduler and KV cache management, the executor and the model runner, and the model code. Based on the vLLM 0.30.0 source, this chapter gives the key files, classes and functions of each segment, mapped one to one onto the mini engine written earlier. Afterwards you should be able to quickly find where any feature lives in the source.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Which process is each of `AsyncLLM`, `EngineCoreClient`, `EngineCoreProc`, `Executor`, `Worker` and `GPUModelRunner` in? How do they communicate?
    2. Why does `SchedulerOutput` split requests into `scheduled_new_reqs` and `scheduled_cached_reqs`?
    3. In vLLM's Qwen2 model code, which optimizations do `qkv_proj`, `gate_up_proj` and `input_layernorm(hidden_states, residual)` correspond to?
    4. How do you run vLLM in a single process to debug it with breakpoints?

??? success "Answers (try first, then expand to compare)"
    1. `AsyncLLM` and `EngineCoreClient` are in the frontend (API server) process; `EngineCoreProc` is a separate EngineCore process (scheduler, KV management, executor); with multiple GPUs, `Executor` starts one worker process per GPU, holding a `Worker` and a `GPUModelRunner`. The frontend and EngineCore communicate over ZMQ, and EngineCore and the workers over a shared-memory message queue.
    2. New requests need their complete information (prompt, sampling parameters, block table) sent to the worker; for requests already running, the worker has cached their state (the persistent batch), so only the increments (newly allocated blocks, new token counts) are sent. Splitting the two means only increments travel, reducing per-step serialization and communication.
    3. `qkv_proj`: the q, k and v projections merged into one matrix multiplication; `gate_up_proj`: gate and up merged into one matrix multiplication; `input_layernorm(hidden_states, residual)`: fused residual addition + RMSNorm (one kernel adds the residual and normalizes).
    4. Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` so EngineCore runs in the same process as the frontend (`InprocClient`); run a small model with the offline `LLM` class, TP=1 and `--enforce-eager`, and you can set breakpoints directly in the scheduler and `GPUModelRunner`.

!!! note "Version"
    The file paths and function names in this chapter are based on vLLM 0.30.0 (September 2026). vLLM moves fast and names may change, but the main structure has been stable since the V1 engine. When reading a newer version, searches such as `grep -rn "def schedule" vllm/v1` will find things again.

## Getting ready {#准备工作}

The fastest way to get the source is to download the source package (no compilation needed):

```bash
pip download vllm==0.30.0 --no-deps --no-binary :all: -d . && tar xzf vllm-0.30.0.tar.gz
# or: git clone https://github.com/vllm-project/vllm && git checkout v0.30.0
```

A few switches are very useful for debugging:

| Setting | Effect |
| --- | --- |
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | EngineCore runs in the same process as the frontend (`InprocClient`), so you can set breakpoints directly in the scheduler |
| `--enforce-eager` | turns off CUDA Graphs and torch.compile, for clearer stack traces while debugging |
| `VLLM_LOGGING_LEVEL=DEBUG` | detailed logs of scheduling, the KV cache and more |
| The offline `vllm.LLM` class | no HTTP server; `llm.generate(...)` goes straight to the engine, the best for stepping through |
| `py-spy dump --pid <pid>` | see where a running process is stuck |

## A map of the directories {#目录地图}

| Directory | Contents |
| --- | --- |
| `vllm/entrypoints/` | the `vllm serve` command line, the OpenAI-compatible server (`openai/api_server.py`, `openai/chat_completion/serving.py`), the offline `LLM` class (`llm.py`) |
| `vllm/v1/engine/` | the frontend engine `AsyncLLM`, input and output processing, `EngineCore` and its multiprocess clients |
| `vllm/v1/core/` | the scheduler (`sched/`), KV cache management (`kv_cache_manager.py`, `block_pool.py`, `kv_cache_utils.py`) |
| `vllm/v1/executor/` | executors: single process, multiprocess, Ray |
| `vllm/v1/worker/` | `Worker` (one per GPU) and `GPUModelRunner`; under `gpu/` is the experimental Model Runner V2 |
| `vllm/v1/attention/` | attention backends (FlashAttention, FlashInfer, MLA, Triton…) and backend selection |
| `vllm/v1/sample/` | the sampler, rejection sampling (speculative decoding), logits processors |
| `vllm/v1/spec_decode/` | draft methods for speculative decoding: EAGLE, n-gram, Medusa, draft models… |
| `vllm/v1/structured_output/` | structured output (xgrammar, guidance and outlines backends) |
| `vllm/model_executor/` | model code (`models/`), parallelized layers (`layers/`: linear layers, MoE, quantization, RoPE…), weight loading |
| `vllm/distributed/` | parallel groups (`parallel_state.py`), communication (`device_communicators/`), KV transfer for PD disaggregation (`kv_transfer/`), EPLB |
| `vllm/compilation/` | torch.compile integration, custom fusion passes, CUDA Graph wrappers |
| `csrc/` | C++/CUDA kernels: paged attention, quantized GEMMs, MoE, custom all-reduce and more |

## Process structure {#进程结构}

![Figure: vLLM's process structure: API server, EngineCore, Worker](../assets/figures/vllm-processes.svg){.aig-svg}

<!-- i18n:diagram 71d5d15c61 -->
```text
API server process                         EngineCore process                           Worker processes (one per GPU)
┌──────────────────────────────┐    ZMQ    ┌──────────────────────────────┐   shared    ┌─────────────────────┐
│ FastAPI routes               │ ────────▶ │ input thread → input_queue   │   memory    │ Worker              │
│ OpenAIServingChat            │  msgpack  │ main thread: run_busy_loop   │   queue     │  GPUModelRunner     │
│ AsyncLLM                     │           │   Scheduler                  │ ──────────▶ │   model + attention │
│  ├ InputProcessor (tokenize) │ ◀──────── │   KVCacheManager             │ (broadcast) │   backend           │
│  └ OutputProcessor (detok.)  │           │   Executor                   │ ◀────────── │   Sampler           │
└──────────────────────────────┘           │ output thread ← output_queue │             └─────────────────────┘
                                           └──────────────────────────────┘
```

- The frontend and EngineCore communicate over ZMQ, with messages encoded in msgpack (`vllm/v1/serial_utils.py`). EngineCore has dedicated input and output threads for sending, receiving and coding, so the main thread only schedules and executes.
- With one GPU, `UniProcExecutor` calls the worker directly inside the EngineCore process; with several, `MultiprocExecutor` (`v1/executor/multiproc_executor.py`) starts one worker process per GPU and broadcasts each step's `SchedulerOutput` to all workers over a shared-memory message queue (`MessageQueue` in `distributed/device_communicators/shm_broadcast.py`).
- Under data parallelism (DP), each DP rank has its own EngineCore (`DPEngineCoreProc`), and the frontend balances load with `DPLBAsyncMPClient`.

## Main line 1: the frontend {#主线一前端}

The docstring of `AsyncLLM.generate()` (`v1/engine/async_llm.py`) sums up the frontend's four jobs:

1. Create an output stream for the request (`RequestOutputCollector`);
2. Process the input: `InputProcessor` renders and tokenizes the prompt into an `EngineCoreRequest`;
3. Register the request with `OutputProcessor` (which handles detokenization, stop strings and assembling `RequestOutput`s);
4. Send the request to the EngineCore process through `EngineCoreClient`.

A background `output_handler` coroutine (`_run_output_handler`) keeps pulling `EngineCoreOutputs` from EngineCore, passes them to `OutputProcessor.process_outputs`, and puts the results into each request's output stream. The HTTP layer (`OpenAIServingChat` in `entrypoints/openai/chat_completion/serving.py`) iterates over this stream to produce the SSE response.

In the mini engine: `AsyncEngine` + `IncrementalDetokenizer` in [`api_server.py`](../engine/sampler-api.md#openai-兼容的流式服务). The difference is that vLLM's engine core is in another process.

## Main line 2: the EngineCore main loop {#主线二enginecore-主循环}

`EngineCoreProc.run_busy_loop()` (`v1/engine/core.py`) is extremely concise:

```py
while self._handle_shutdown():
    self._process_input_queue()      # 1) take new requests, aborts, etc. (block when there is no work)
    self._process_engine_step()      # 2) run one step and put the outputs in output_queue
```

A step's content is in `EngineCore.step()`:

```py
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)   # structured output mask, computed in parallel with the forward pass
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```

Note that the forward pass (`execute_model`) and sampling (`sample_tokens`) are two separate calls: this lets the scheduler compute the grammar mask for structured output on the CPU while the GPU runs the forward pass.

For pipeline parallelism and asynchronous scheduling, `step_with_batch_queue()` is used: it keeps a queue of batches, **first scheduling and submitting new batches as far as possible, then waiting for the earliest batch to return**, so the GPU does not wait for CPU scheduling. Asynchronous scheduling (`--async-scheduling`, on by default when the conditions allow) is implemented by `AsyncScheduler` (`v1/core/sched/async_scheduler.py`): the next step's scheduling does not wait for the previous step's sampled results; it first puts a "placeholder" token for each request (`num_output_placeholders`) and fills it in when the results come back.

In the mini engine: `LLMEngine.step()`.

## Main line 3: the scheduler and KV cache management {#主线三调度器与-kv-cache-管理}

The scheduler's algorithm was compared in detail in [the scheduler chapter](../engine/scheduler.md); here are the data structures:

- **`Request`** (`v1/request.py`): a request's state in EngineCore, including `num_computed_tokens`, `spec_token_ids`, `block_hashes` and the state machine `RequestStatus`.
- **`SchedulerOutput`** (`v1/core/sched/output.py`): the scheduling result. New requests are sent in full as `NewRequestData` (prompt tokens, sampling parameters, block IDs); requests already running send only **increments** as `CachedRequestData` (newly allocated blocks, new tokens, `num_computed_tokens`). The worker keeps a "persistent batch" (`InputBatch`) that only needs updating by the increments. So very little data crosses processes each step.
- **`KVCacheManager`** (`v1/core/kv_cache_manager.py`): `get_computed_blocks` looks up the prefix cache, `allocate_slots` allocates blocks for this step, and `free` releases them. Below it are `KVCacheCoordinator` and the `SingleTypeKVCacheManager` subclasses by attention type (`FullAttentionManager`, `SlidingWindowManager`, `MambaManager`…): models with hybrid architectures (say, some layers with sliding windows and some with linear attention) are split into several **KV cache groups**, each with its own management rules.
- **`BlockPool`** (`v1/core/block_pool.py`): physical blocks, the free queue, and the mapping from hashes to blocks; see [the prefix caching chapter](../engine/prefix-cache.md).

`update_from_output()` runs after the forward pass: it appends the sampled tokens to the requests, handles tokens rejected in speculative decoding, checks stop conditions (EOS, maximum length, stop tokens), frees the blocks of finished requests, and packs the results into `EngineCoreOutputs` grouped by frontend.

## Main line 4: Worker and GPUModelRunner {#主线四worker-与-gpumodelrunner}

The lifecycle of a `Worker` (`v1/worker/gpu_worker.py`):

| Method | What it does |
| --- | --- |
| `init_device` | set up the GPU and initialize the distributed environment (NCCL groups) |
| `load_model` | build the model and load the weights (`model_executor/model_loader/`) |
| `determine_available_memory` | run a "profiling" forward pass with the largest batch (`profile_run`) to measure peak memory; what is left (scaled by `gpu_memory_utilization`) goes to the KV cache |
| `initialize_from_config` | allocate the KV tensors from the KV cache config (`initialize_kv_cache`) |
| `compile_or_warm_up_model` | torch.compile and CUDA Graph capture (`capture_model`) |

At each step, `GPUModelRunner.execute_model()` (`v1/worker/gpu_model_runner.py`) does the following:

1. `_update_states`: update the persistent batch from the increments in `SchedulerOutput` (add new requests, remove finished ones, update block tables);
2. `_prepare_inputs`: build `input_ids`, `positions`, `query_start_loc`, `slot_mapping` and `logits_indices`;
3. Build the attention metadata (each backend has its own metadata builder), and choose the CUDA Graph mode and padded size;
4. Run the model's forward pass inside `set_forward_context(...)`. Attention layers do not receive the metadata as arguments but read it from this "forward context", so the model code need not care about the batch structure;
5. Compute logits only for `logits_indices`.

Then `sample_tokens()` samples (and, for speculative decoding, verifies and generates the next round's draft).

In the mini engine: `build_batch` (step 2) and `ModelRunner.forward` (steps 4 and 5).

## Main line 5: the model code {#主线五模型代码}

The attention and decoder layers of the Qwen2 model in vLLM (`model_executor/models/qwen2.py`) look like this (excerpt, Apache-2.0 license):

```py
class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, quant_config=None, prefix=""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False, ...)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, ...)
        self.act_fn = SiluAndMul()

class Qwen2Attention(nn.Module):
    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

class Qwen2DecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

Compared with `mini_llm` in the LLM book, every difference is an inference optimization:

| mini_llm | vLLM | Optimization |
| --- | --- | --- |
| three linear layers `q_proj`, `k_proj`, `v_proj` | one `QKVParallelLinear` whose output is then `split` | three GEMMs merged into one; split by head under tensor parallelism |
| `gate_proj`, `up_proj` + `silu(g) * u` | `MergedColumnParallelLinear` + `SiluAndMul` | two GEMMs merged; the activation and multiplication fused into one kernel |
| `x = x + attn(norm(x))` | `input_layernorm(hidden_states, residual)` returns both the normalized result and the new residual | residual addition fused with RMSNorm (fused add RMSNorm) |
| `o_proj`, `down_proj` as ordinary linear layers | `RowParallelLinear` | split by rows under tensor parallelism, followed by an all-reduce |
| input `[B, T]`, concatenated contiguous KV cache | one-dimensional tokens + `positions` as input; attention reads metadata from the forward context and writes paged KV | variable-length batching, paged KV |
| `rope_cos_sin` computed every time | `get_rope` returns a RoPE module with a precomputed cos/sin cache | cached cos/sin, fused kernels |

The `@support_torch_compile(dynamic_arg_dims={"input_ids": {0: "b"}, ...})` decorator on the model class declares which dimension is the dynamic token count; this is the "compile once with a symbolic token count" of [the previous chapter](../engine/graphs-compile.md). The implementations of these parallel layers are covered in [the tensor parallelism chapter](../distributed/tensor-parallel.md).

## Other modules at a glance {#其他模块速查}

| To learn about | Look at |
| --- | --- |
| Choosing the attention backend | `v1/attention/selector.py`, with the backends in `v1/attention/backends/` |
| Speculative decoding | `v1/spec_decode/eagle.py` (EAGLE/MTP drafts), `ngram_proposer.py`, with verification in `v1/sample/rejection_sampler.py` |
| Structured output | `v1/structured_output/`; the grammar mask is computed in EngineCore (`get_grammar_bitmask`) |
| PD disaggregation, KV offloading | `distributed/kv_transfer/` (the KV connector interface and implementations such as NIXL and LMCache), `v1/kv_offload/` |
| MoE and expert parallelism | `model_executor/layers/fused_moe/`, `distributed/device_communicators/all2all.py`, `distributed/eplb/` |
| Quantization | `model_executor/layers/quantization/` |
| LoRA | `vllm/lora/`, and `lora_model_runner_mixin.py` on the worker side |
| Multimodality | `vllm/multimodal/`, and the encoder cache in the scheduler (`v1/core/encoder_cache_manager.py`) |
| Parallel groups | `distributed/parallel_state.py` (setting up the TP, PP, DP and EP groups) |

## A suggested reading order {#建议的阅读顺序}

1. Run a small model with the offline `LLM` class, `VLLM_ENABLE_V1_MULTIPROCESSING=0` and `--enforce-eager`, set breakpoints in `Scheduler.schedule` and `GPUModelRunner.execute_model`, and step through once;
2. Read `EngineCore.step` → `Scheduler.schedule` → `update_from_output`, against this handbook's scheduler chapter;
3. Read `KVCacheManager` and `BlockPool`, against the paged KV and prefix caching chapters;
4. Read `GPUModelRunner._prepare_inputs` and one attention backend (starting from `flash_attn.py`), against the variable-length batching chapter;
5. Read one model file (`qwen2.py` or `llama.py`) and `layers/linear.py`, against the tensor parallelism chapter;
6. Go deeper into topics as you like: speculative decoding, PD disaggregation, MoE.

!!! interview "How to explain it"
    To explain "describe vLLM's architecture", going by process is clearest: the **frontend process** (HTTP, tokenization, detokenization, `AsyncLLM`), the **EngineCore process** (a busy loop: schedule → execute → update, `Scheduler` + `KVCacheManager`), and the **worker processes** (one per GPU, `GPUModelRunner` runs the model). The processes communicate over ZMQ and a shared-memory queue respectively, and scheduling results carry only increments. Then talk about one or two modules you have read in depth (for example the scheduler's unified token budget, or the LRU design of the KV block pool), which is more convincing than generalities.

## Exercises {#练习}

**1. Find a parameter.** Where does `max_num_seqs` take effect? Find it in the source and explain what it and `max_num_batched_tokens` each limit.

??? success "Answer"
    It is read in `Scheduler.__init__` as `self.max_num_running_reqs`, and when `schedule()` schedules the waiting queue it stops admitting new requests once `len(self.running) + ... >= self.max_num_running_reqs`; it limits **the number of requests running at once** (and also determines things like the largest CUDA Graph capture size). `max_num_batched_tokens` is the `token_budget` at the start of `schedule()`, limiting **the total number of tokens computed in one step**.

**2. Add a log line.** To print "how many requests and tokens were scheduled this step, and KV cache usage" at every step, where would you add it?

??? success "Answer"
    In `EngineCore.step()` after `schedule()`: `len(scheduler_output.num_scheduled_tokens)` is the number of requests, `scheduler_output.total_num_scheduled_tokens` the number of tokens, and `self.scheduler.get_kv_cache_usage()` the usage. In fact vLLM already collects these metrics through `make_stats()` and exposes them in Prometheus format on the `/metrics` endpoint (`vllm:num_requests_running`, `vllm:kv_cache_usage_perc` and so on), so in production you just look at the monitoring.

## Summary {#小结}

- [x] vLLM V1 has three kinds of processes: the frontend (`AsyncLLM`), EngineCore (scheduling + KV management + executor) and workers (`GPUModelRunner`), communicating over ZMQ and a shared-memory queue respectively.
- [x] The main loop is `schedule → execute_model → sample_tokens → update_from_output`; asynchronous scheduling and the batch queue overlap CPU scheduling with GPU execution.
- [x] `SchedulerOutput` carries only increments, and the worker keeps a persistent batch; attention layers read metadata from the forward context.
- [x] Every merged projection, fused operator and parallel linear layer in the model code corresponds to an inference optimization.
