# Architecture and the path we follow

<p class="lead">Before writing anything, look at the whole system once: a request arrives over HTTP, passes through a set of processes, turns into a set of data structures, gets computed somewhere, and finally flows back to the client one token at a time. This chapter gives the whole picture of mini-sglang, the dependencies between its modules, the order in which we rebuild it, and how to verify every step on a machine with no GPU.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What kinds of processes make up mini-sglang? How many processes does a TP=4 service have in total?
    2. Inside a scheduler process, what does the "scheduler" do and what does the "engine" do?
    3. Why does the model's `forward()` take no arguments?
    4. The official implementation only supports CUDA. Which three means does this book use to verify GPU-related code on a CPU?

??? success "Answers (try it yourself first, then expand)"
    1. Three kinds: the API server (the HTTP entry point), the tokenizer / detokenizer process, and one scheduler process per TP rank (each holding a scheduler and an engine). At TP=4 that is 1 + 1 + 4 = 6 processes.
    2. The scheduler decides what to compute this step and where the KV goes: it receives requests, picks the batch, allocates KV and handles the results. The engine does the computing: it runs the model's forward pass and the sampling.
    3. Everything about the current batch (requests, positions, attention metadata) lives in a global `Context`, and each model layer reads it from there when it needs it, so `forward()` needs no arguments.
    4. The device abstraction (the same code runs on a CPU), reference implementations and stand-ins with the same interface as the GPU ones (the attention backend, the fake FlashInfer / FlashAttention), and simulators (the CUDA Graph emulation, the CPU simulator for CUDA kernels, the Triton interpreter).

## The result first {#先看结果}

This is what we are going to build: the offline `LLM` interface running Qwen3-0.6B on a CPU, generating for three requests at once.

@@code examples/ch00_quickstart.py@@

@@output ch00_quickstart@@

The online service is a single command, `python -m minisgl --model Qwen/Qwen3-0.6B`, after which any OpenAI client can talk to `http://127.0.0.1:1919/v1/chat/completions`.

## The processes {#进程结构}

mini-sglang is a multi-process system. At TP=2 it looks like this:

@@diagram processes mini-sglang's processes (TP=2)@@

- **API server** (the main process): takes HTTP requests, gives each one a `uid`, and sends the text to the tokenizer; it then streams the delta text it gets back to the client over SSE.
- **tokenizer / detokenizer**: tokenization and incremental detokenization. By default both share one process (`--num-tokenizer 0`).
- **scheduler processes**: one per TP rank. Each holds a `Scheduler` (which decides which requests run this step and allocates KV cache) and an `Engine` (which owns the model and the KV pool and runs the forward pass and sampling). Only rank 0 talks to the tokenizer; it broadcasts every message it receives verbatim to the other ranks, so all ranks make exactly the same scheduling decisions.

So a TP=4 service has 1 + 1 + 4 = 6 processes in total (API server, tokenizer, four schedulers). Control messages between processes go over ZMQ; tensor-parallel data goes over NCCL.

!!! upstream "The official implementation"
    The startup logic is in @@upstream server/launch.py:launch_server@@ and the scheduler's main loop in @@upstream scheduler/scheduler.py:Scheduler.overlap_loop@@. The official [docs/structures.md](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/docs/structures.md) has a process diagram that matches the one above.

## The journey of one request {#一个请求的旅程}

@@diagram request-lifecycle the processes and messages a request passes through@@

1. The client sends `POST /v1/chat/completions`. The API server assigns `uid = 7` and sends `TokenizeMsg(uid=7, text=[messages], sampling_params)`.
2. The tokenizer applies the chat template, tokenizes, and sends `UserMsg(uid=7, input_ids=tensor)` to scheduler rank 0.
3. The scheduler puts it in the prefill waiting queue. On some scheduling step `PrefillAdder` looks up the longest cached prefix in the radix cache, checks whether there is enough memory left and a free request slot, and admits it if so: it allocates a page-table row and locks the prefix it hit.
4. `_prepare_batch` allocates KV pages for the tokens to compute this step, works out each token's position and the KV location to write, and the attention backend prepares its metadata.
5. `Engine.forward_batch`: run the model forward, take the logits at each request's last position, and sample. The sample is written straight back into the `token_pool` on the GPU, as the next step's input, and is also copied back to the CPU asynchronously.
6. The scheduler handles the result: it appends the new token to the request, decides whether it has finished, and sends `DetokenizeMsg(uid=7, next_token, finished)`. For a request that has just finished prefill, the prompt's KV is inserted into the radix cache.
7. Every decode step after that repeats steps 4 to 6 until EOS or `max_tokens`. At the end the request slot is freed and the KV handed back to the cache.
8. The detokenizer turns tokens into text incrementally and sends `UserReply(uid=7, incremental_output, finished)`; the API server writes it into the SSE stream.

Every step along this path is a chapter of this book.

@@video lifecycle Animation: the life of a request (about 2 minutes, Chinese narration and subtitles)@@

## Modules and dependencies {#模块与依赖}

The modules under `python/minisgl/`, from the bottom of the dependency stack up:

@@diagram modules module layering@@

| Module | What it holds | Chapter |
| --- | --- | --- |
| `utils`, `env` | logging, the registry, ZMQ queues, HF helpers; (ours) the device abstraction | introduced where first used |
| `core` | `SamplingParams`, `Req`, `Batch`, `Context` | [Core data structures](../compute/core.md) |
| `distributed` | TP info, all-reduce / all-gather | [Tensor parallelism](../perf/tensor-parallel.md) |
| `kernel` | ops: RMSNorm, RoPE, KV writes, lookups (CPU reference plus GPU versions) | [Op layer](../compute/layers.md), [CUDA kernels](../perf/kernels.md) |
| `layers` | the `BaseOP` system and the layers | [Op layer](../compute/layers.md) |
| `models` | model config, model structure, weight loading | [Model and weight loading](../compute/models.md) |
| `kvcache` | the KV pool and the prefix cache (naive and radix) | [KV pool](../compute/kvcache.md), [Radix cache](../schedule/radix-cache.md) |
| `attention` | attention backends: torch (ours), FlashInfer, FlashAttention | [Attention backend](../compute/attention.md), [GPU attention](../perf/gpu-attention.md) |
| `moe` | MoE backends: the torch reference, the Triton fused MoE | [MoE](../perf/moe.md) |
| `engine` | `Engine`, `Sampler`, `GraphRunner` | [Engine and sampler](../compute/engine.md), [CUDA Graph](../perf/cuda-graph.md) |
| `scheduler` | the scheduler and its managers | all of part three |
| `message` | inter-process messages and serialization | [Messages and ZMQ](../serve/message.md) |
| `tokenizer` | the tokenizer / detokenizer process | [Tokenizer](../serve/tokenizer.md) |
| `server`, `llm` | the API server and launcher; the offline `LLM` interface | [API server](../serve/api-server.md), [Scheduler skeleton](../schedule/scheduler.md) |

Two designs run through everything and are worth remembering now:

- **The global `Context`.** The model's `forward()` has no arguments: the input tokens, the positions and the attention metadata all hang off "the current batch", and the current batch lives in a process-wide global `Context`. Any layer that needs them calls `get_global_ctx().batch`. This keeps the model code very clean, at the price that the model only runs inside the `ctx.forward_batch(batch)` context.
- **The `Registry`.** `--attn fa,fi`, `--cache-type radix` and `--moe-backend fused` on the command line are names, and the registry maps them to implementations. Adding a backend means writing one class and registering one name.

## The path we follow {#复刻路线}

The book goes in the order "compute it right, then schedule it well, then serve it, then make it fast and big":

1. **Computing it right** (chapters 1-6): no scheduler, batches assembled by hand, the model prefilling and decoding correctly on a paged KV cache. The finish line is a hand-written loop whose greedy output matches HF token by token.
2. **Scheduling it well** (chapters 7-11): add the scheduler. First the plainest continuous batching, then admission control, the radix cache, chunked prefill and overlap scheduling. The output must not change at any step.
3. **Serving** (chapters 12-15): messages, the tokenizer process, the scheduler's IO and the API server, giving a complete online service.
4. **Faster and bigger** (chapters 16-21): tensor parallelism, GPU attention backends, CUDA Graph, custom kernels, MoE and benchmarks.

Every chapter has the same shape: self-test, the idea, our implementation ("the files you will write"), a comparison with the official code, tests and what they print, exercises, summary.

## Verifying GPU code on a CPU {#在-cpu-上验证-gpu-代码}

Having no GPU on your development machine is the normal case. There are four kinds of CUDA-only things in the official code, and the book handles each:

| What upstream uses | What this book does | Chapter |
| --- | --- | --- |
| `torch.cuda.Stream`, `Event`, pinned memory | `utils/device.py`: used as is on CUDA, replaced by do-nothing `NullStream` and `NullEvent` on CPU | [Engine](../compute/engine.md), [Overlap scheduling](../schedule/overlap.md) |
| FlashInfer / FlashAttention attention kernels | an extra PyTorch reference backend `torch`; in tests, a fake FlashInfer / fake sgl_kernel with the same interface checks the argument semantics of the official backend code | [Attention backend](../compute/attention.md), [GPU attention](../perf/gpu-attention.md) |
| CUDA Graph | `EmulatedGraph` on the CPU: replay can only read the fixed buffers, so missing any input copy makes the result wrong | [CUDA Graph](../perf/cuda-graph.md) |
| custom CUDA / Triton kernels | CUDA kernels are compiled with nvcc 12.9 and 13.4 and self-checked on the CPU simulator from the [CUDA handbook](cuda://); Triton kernels run in interpreter mode | [CUDA kernels](../perf/kernels.md), [MoE](../perf/moe.md) |

On a GPU the same code picks FlashInfer / FlashAttention by itself (`--attn auto`), turns on CUDA Graph, and uses the custom kernels and the Triton fused MoE.

!!! diff "Difference from upstream: the device abstraction"
    The first thing the official `Engine.__init__` does is `torch.cuda.set_device`, and the scheduler creates a `torch.cuda.Stream()` directly. We replace those with `create_stream(device)`, `create_event(device)`, `pin(device)` and friends, which behave identically on CUDA. `EngineConfig` also gains two fields, `device` and `cpu_kv_cache_bytes`.

## Environment {#环境}

```bash
git clone https://github.com/AnranS/ai-infra-handbooks && cd ai-infra-handbooks/minisgl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # CPU torch is enough; on a GPU also install ".[gpu]"
# model: put it under models/ (we run float32, which matches HF most reliably)
modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B
```

It is worth cloning the official repository next to it and reading along:

```bash
git clone https://github.com/sgl-project/mini-sglang && git -C mini-sglang checkout 9a91cfa
```

!!! info "Running it on a Mac"
    None of the code in this book requires a GPU: without CUDA the `Engine` picks the CPU, and `torch.cuda.Stream` / `Event` / pinned memory all degrade to no-ops (see `utils/device.py`); FlashInfer and FlashAttention are replaced by same-interface PyTorch fakes, and CUDA Graph by `EmulatedGraph`. On macOS, install the dependencies and `PYTHONPATH=python:tests pytest -q tests` just runs — the tensor-parallel chapters use gloo and are still genuinely multi-process. Two caveats:

    - **MPS is not used.** `_pick_device` only recognises CUDA; everything else is the CPU. That is deliberate — every test here compares against Hugging Face token by token in float32, and MPS is neither precise nor complete enough for that.
    - **`tools/check.py` is not for readers.** It is the maintainer's full check and needs the upstream repository at a pinned commit plus two nvcc toolchains. `pytest` is all you need.

!!! warning "Thread count on a CPU"
    PyTorch on a CPU defaults to one thread per logical core. The matmuls during decode are tiny, with a batch of only a few, and synchronizing 32 threads makes them tens of times slower: on this book's development machine one decode step takes 2.4 seconds with the default and 40 milliseconds once the thread count drops to the physical core count. So our `Engine` sets the thread count to `physical cores / TP size` on a CPU, unless you set `OMP_NUM_THREADS` yourself.

!!! interview "How to explain it"
    To explain how to "describe the architecture of an inference engine", you can use mini-sglang as the skeleton: an API server (HTTP, the OpenAI interface) plus a tokenizer / detokenizer process plus one scheduler process per TP rank, with control messages over ZMQ and tensors over NCCL, so 1 + 1 + 4 processes at TP=4. Inside a scheduler process the scheduler decides which requests run this step and where the KV goes, and the engine does the computing (model, attention backend, sampling, CUDA Graph); the model reads the current batch from a global `Context`, which is why `forward()` takes no arguments. Then walk through one request (tokenize, queue, prefill, decode step by step, detokenize and stream back, free the KV), and give the reason for multiple processes: it gets around the GIL and lets CPU work run alongside GPU compute.

## Exercises {#练习}

**1. Count the processes.** You deploy a TP=8 service with mini-sglang. How many processes are there? Which pairs exchange control messages and which exchange tensors? Over what?

??? success "Answer"
    One API server + one tokenizer / detokenizer process + eight scheduler processes (one per TP rank, each holding a scheduler and an engine), so ten. Control messages such as requests, tokenized requests and generated results go over ZMQ: API server to tokenizer to scheduler rank 0, with rank 0 forwarding the raw message to the other ranks, and results travelling back from rank 0 through the detokenizer to the API server. Tensors between the eight scheduler processes (the tensor-parallel all-reduces and so on) go over NCCL.

**2. Why this order.** The path is "compute it right, schedule it well, serve it, make it fast and big". Why not write CUDA Graph and the custom kernels first?

??? success "Answer"
    A performance optimization does not change the result, only how it is computed, so its correctness can only be judged against a version that is already correct: first get the reference implementation matching HF token by token, then check every added optimization (overlap scheduling, CUDA Graph, FlashInfer, custom kernels) against the same greedy output. Do performance first and a wrong answer could be the model, the scheduling or the optimization itself. Scheduling and serving also set the boundaries for the optimizations: CUDA Graph needs its inputs in fixed buffers and overlap scheduling needs the input token to stay on the GPU, and both are much easier once the data structures are settled.

## Summary {#小结}

- [x] mini-sglang = API server + tokenizer/detokenizer + one scheduler process per TP rank; control messages over ZMQ, tensors over NCCL.
- [x] The scheduler decides what runs this step and where the KV goes, the engine does the computing, and the model reads the current batch from the global `Context`.
- [x] The order: compute it right, schedule it well, serve it, make it fast and big, with every step matching HF token by token.
- [x] GPU-only parts are verified on a CPU through the device abstraction, reference implementations, same-interface stand-ins and simulators.
