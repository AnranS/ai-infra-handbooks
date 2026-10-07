# Profiling inference engines

<p class="lead">Load testing tells you "it's slow"; profiling tells you "where it's slow". An inference engine's performance problems spread over many layers: CPU overhead in scheduling and input preparation, gaps between kernels on the GPU, an inefficient kernel, communication, needless synchronization. This chapter gives a top-down method for locating them, actually locates a bottleneck in the mini engine with the PyTorch profiler, then introduces the tools for profiling vLLM and SGLang on GPUs and the common symptoms.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Given the problem "TPOT is twice as slow as expected", in what order do you investigate?
    2. On a Nsight Systems timeline, what do recurring blank stretches on the GPU usually mean?
    3. How do you get vLLM or SGLang to output a torch profiler trace?
    4. Why can a single `.item()` or `.tolist()` noticeably lower throughput?

??? success "Answers (try first, then expand to compare)"
    1. Top down: first the end-to-end metrics and the theoretical lower bound (how far off), → server-side metrics (queueing, batch size, KV usage) → the timeline of one step (CPU gaps, communication, which kinds of operators dominate) → only then individual kernels.
    2. The GPU is waiting for the CPU: kernels too small and launches can't keep up, or there is a synchronization in between (`.item()`, a copy), or the CPU is busy scheduling and preparing metadata. The fixes are CUDA Graphs, fusion, removing syncs, and overlapping the CPU's work with GPU compute.
    3. vLLM: start with `--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/abs/path"}'`, then `POST /start_profile` and `POST /stop_profile`; SGLang: set `SGLANG_TORCH_PROFILER_DIR` and call `/start_profile` and `/stop_profile` the same way, or add `--profile` to the load-test command. Open the resulting trace in Perfetto.
    4. They copy a value from the GPU back to the CPU, which must wait for all previously submitted GPU work to finish: the CPU stops to wait, then the GPU, once done, waits for the CPU to submit the next step; the pipeline breaks, and each side idles waiting for the other.

## The top-down method {#自顶向下的方法}

1. **End-to-end metrics**: first confirm the problem with a load test (TTFT or TPOT? slow under every load, or only under high load?), and compare with the theoretical lower bound ([the lower bound on latency](llm://inference/estimation/#延迟的下限) in the LLM handbook): whether measured TPOT is 1.2× or 3× the bound decides where to look next.
2. **Server-side metrics**: queued requests, KV Cache usage, preemption counts, prefix cache hit rate, batch size per step. A lot of "slowness" is really a scheduling or capacity problem that has nothing to do with kernels.
3. **The timeline of one step**: use a profiler to see the phases of one decode step: scheduling and input preparation on the CPU, the forward pass on the GPU, sampling, result processing. Is the GPU busy all the time?
4. **Operators and kernels**: which kernels take most of the time? How far is each from its roofline?

![Figure: finding the bottleneck top down: end-to-end metrics, server-side metrics, one step's timeline, kernels](../assets/figures/top-down.svg){.aig-svg}

## Locating a bottleneck in the mini engine {#在迷你引擎上定位瓶颈}

Label several phases of one engine step with `record_function` (without modifying the engine's code, just wrapping it), then record 10 decode steps with the PyTorch profiler:

```python
import functools
import os
import tempfile
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer
from mini_llm import Transformer
import nano_engine
import runner
from nano_engine import LLMEngine, SamplingParams

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

def traced(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with record_function(name):
            return fn(*args, **kwargs)
    return wrapper

engine = LLMEngine(model, eos_token_id=tok.eos_token_id)
engine.scheduler.schedule = traced("1 调度", engine.scheduler.schedule)
nano_engine.build_batch = traced("2 构造批次", nano_engine.build_batch)
engine.runner.forward = traced("3 前向", engine.runner.forward)
engine.sample_fn = traced("4 采样", engine.sample_fn)
for i in range(16):
    engine.add_request(tok(f"第{i}个问题：介绍一种水果。").input_ids, SamplingParams(max_tokens=40, ignore_eos=True))
for _ in range(3):                                          # run a few steps first: prefill and warm-up are not counted
    engine.step()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(10):
        engine.step()

stats = {e.key: e for e in prof.key_averages()}
for name in ("1 调度", "2 构造批次", "3 前向", "4 采样"):
    print(f"{name}：每步 {stats[name].cpu_time_total / 10 / 1000:6.2f} ms")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=6))
prof.export_chrome_trace(os.path.join(tempfile.gettempdir(), "nano_engine_trace.json"))   # open it at ui.perfetto.dev
```

```text
1 调度：每步   0.03 ms
2 构造批次：每步   0.09 ms
3 前向：每步 241.65 ms
4 采样：每步   0.40 ms
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   aten::mm        24.56%     594.859ms        24.59%     595.573ms     302.321us          1970  
                       3 前向        19.75%     478.372ms        99.78%        2.416s     241.649ms            10  
                aten::copy_         8.28%     200.510ms         8.28%     200.510ms       8.118us         24700  
                aten::index         5.86%     141.849ms         6.46%     156.465ms      17.443us          8970  
                  aten::bmm         5.30%     128.391ms         5.39%     130.446ms      14.559us          8960  
               aten::einsum         4.32%     104.613ms        16.53%     400.300ms      44.676us          8960  
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.422s
```

Almost all the time is in the forward pass, and scheduling and input preparation are negligible (on a GPU the proportions are completely different; see below). But the operator table inside the forward pass reveals a problem: `aten::einsum`, `aten::bmm` and `aten::index` are called thousands of times, far more than the matrix multiply `aten::mm`. This is the reference implementation written in the [paged KV chapter](../engine/paged-kv.md#分页注意力): attention loops separately over **each request** in the batch. Label attention on its own and see how it changes with batch size:

```python
runner.paged_attention = traced("paged_attention", runner.paged_attention)
for batch in (4, 16, 64):
    engine = LLMEngine(model, eos_token_id=tok.eos_token_id, max_num_seqs=64, num_blocks=1024)
    engine.runner.forward = traced("forward", engine.runner.forward)
    for i in range(batch):
        engine.add_request(tok(f"第{i}个问题：介绍一种水果。").input_ids, SamplingParams(max_tokens=40, ignore_eos=True))
    for _ in range(3):
        engine.step()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(5):
            engine.step()
    stats = {e.key: e for e in prof.key_averages()}
    forward, attention = stats["forward"].cpu_time_total, stats["paged_attention"].cpu_time_total
    print(f"batch {batch:2d}：每步 {forward / 5 / 1000:5.0f} ms，其中注意力占 {attention / forward:.0%}，"
          f"每步 einsum 调用 {stats['aten::einsum'].count // 5} 次")
```

```text
batch  4：每步    94 ms，其中注意力占 41%，每步 einsum 调用 224 次
batch 16：每步   237 ms，其中注意力占 62%，每步 einsum 调用 896 次
batch 64：每步   715 ms，其中注意力占 81%，每步 einsum 调用 3584 次
```

Attention's call count is proportional to batch size, and its share of a step grows from 40% to over 80%. This is why real inference engines must use paged attention with **one kernel for the whole batch** (FlashAttention's varlen interface, FlashInfer, vLLM's paged attention kernel): one launch, computing every request in parallel internally by its block table. It is also a concrete example of the launch overhead problem described in the CUDA Graphs chapter.

The `trace.json` the profiler exports can be opened in Perfetto to see the timeline of each phase and operator in every step, which is more intuitive than the table.

## On GPUs: tools {#在-gpu-上工具}

| Tool | What it shows | Usage |
| --- | --- | --- |
| PyTorch profiler | Python call stacks + CUDA kernels, as a timeline | vLLM: `--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/abs/path"}'`, then `POST /start_profile`, `POST /stop_profile`; SGLang: set `SGLANG_TORCH_PROFILER_DIR` and call `/start_profile` and `/stop_profile`, or add `--profile` to the load-test command |
| Nsight Systems (nsys) | the whole system's timeline: CPU threads, CUDA API, kernels, memory copies, NCCL, NVTX markers | `nsys profile -t cuda,nvtx,osrt --trace-fork-before-exec=true --cuda-graph-trace=node -o report vllm serve ...` (multiple processes need fork tracing, and kernels inside CUDA Graphs need expanding per node) |
| Nsight Compute (ncu) | the details of a single kernel: bandwidth, compute utilization, occupancy, hit rates at each cache level, the roofline | sample and analyze selected kernels; see [performance analysis: Nsight](cuda://tools/profiling/) in the CUDA handbook |

## Common symptoms and causes {#常见症状与原因}

| Symptom | Common causes | Direction |
| --- | --- | --- |
| Many gaps between kernels on the GPU timeline, especially with small batches | CPU launch overhead, slow scheduling and input preparation | CUDA Graphs, asynchronous scheduling, less Python overhead ([CUDA Graphs](../engine/graphs-compile.md)) |
| The GPU idles for a while at the start of every step | input preparation does not overlap with GPU compute; H2D copies without pinned memory or non_blocking | asynchronous scheduling, persistent batches, pinned buffers |
| A gap after sampling or result processing | CPU-GPU syncs from `.item()`, `.tolist()`, `torch.multinomial` and the like | move logic that depends on the results to the next step, or onto the GPU |
| Attention's share of decode steps keeps growing with context | reading KV is the bottleneck (this is normal), or the decode kernel does not split KV for long contexts | check the attention backend; KV quantization; FlashDecoding-style splitting |
| Noticeable NCCL all-reduce time in every layer | TP communication latency (small messages) | custom all-reduce, fusing all-reduce with RMSNorm, adjusting TP ([tensor parallelism](../distributed/tensor-parallel.md#通信的代价)) |
| All-to-all and expert compute run serially in MoE layers | EP communication is not overlapped | two-batch overlap, DeepEP's low-latency mode ([expert parallelism](../distributed/expert-parallel.md)) |
| Some steps are especially slow, correlated with prefills | long prefills batched together with decode | lower the chunked prefill budget, PD disaggregation |
| GEMMs take far longer than the roofline estimate | bad shapes (a batch so small it becomes GEMV), Tensor Cores unused, an unsuitable quantization kernel | check kernel names and shapes; switch to a better-suited quantization scheme |

!!! source "Source code"
    - **vLLM**: the profiler configuration is in `vllm/config/profiler.py` (`ProfilerConfig`, with `torch`, `cuda` and `proton` options), and the HTTP interface in `vllm/entrypoints/serve/profile/`; `vllm/profiler/layerwise_profile.py` can break time down per layer. The code is full of labels like `record_function_or_nullcontext("schedule: allocate_slots")`, so the trace shows the time of each scheduler phase directly.
    - **SGLang**: `srt/managers/scheduler_components/profiler_manager.py` starts and stops the profiler, and `SGLANG_TORCH_PROFILER_DIR` sets the output directory; `sglang.benchmark.one_batch` can profile a single batch without starting the server.

!!! interview "In an interview"
    When asked "how do you do performance optimization", the best answer is a concrete story told as "symptom → hypothesis → tool → evidence → change → result": for example, "TPOT was twice the theoretical bound → suspected CPU overhead → nsys showed gaps between kernels and 3 ms of GPU idle at the start of each step → traced it to a Python loop and a sync in input preparation → vectorized it and removed the sync → TPOT dropped 40%". Citing the roofline bound as a reference and naming specific trace features makes you sound very solid. This chapter's mini-engine example also works as a small story.

## Exercises {#练习}

**1. Reading a trace.** On an nsys timeline you see that in every decode step the GPU's kernels are packed tightly with almost no gaps, yet the total time is still 1.8× the theoretical bound. What do you check next?

??? success "Approach"
    The GPU is busy all the time, so it is not CPU overhead; the kernels themselves are slow. Sort kernels by time: if GEMM/GEMV dominates, use ncu to see their actual bandwidth utilization (decode GEMVs should be near 70%–90% of peak bandwidth) and check that suitable kernels are used (quantization format, split-K); if attention dominates, check the context length and batch size, compute the theoretical time to read the KV, and see how far the attention kernel is from it; then look for unexpected kernels (extra copies, type conversions, unfused elementwise operations).

**2. Where is the sync?** Why does calling `next_tokens.tolist()` after sampling hurt throughput? How does vLLM avoid this?

??? success "Answer"
    `.tolist()` must copy data from the GPU back to the CPU, which waits for all previously submitted GPU work to finish (a sync). During that time the CPU can do nothing, and cannot prepare the next step's inputs in advance; after the copy, the GPU must wait for the CPU to prepare the next step before it can start, and the two take turns idling. vLLM's asynchronous scheduling has the scheduler schedule the next step with placeholder tokens and prepare its inputs ahead of time, filling in the sampled results when they arrive; on the worker side, pinned memory and non_blocking copies overlap returning results with the next step's compute. SGLang's overlap scheduling does the same thing.

## Summary {#小结}

- [x] Top down: end-to-end metrics and the theoretical bound → server-side metrics → one step's timeline → individual kernels.
- [x] Label engine phases with `record_function`; the profiler's operator table and trace expose problems quickly (in this chapter: per-request attention loops slow down linearly with batch size).
- [x] On GPUs, use the torch profiler (both vLLM and SGLang have built-in switches), Nsight Systems for timelines, and Nsight Compute for individual kernels.
- [x] Common problems: CPU overhead and gaps, syncs, KV reads, TP/EP communication, prefill interference, poorly chosen kernels, each with its own direction for a fix.
