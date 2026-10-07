# Project: the GPU performance bar for an inference engine

<p class="lead">This book had you implement and verify every module of mini-sglang on a CPU. The project asks you to move it to a GPU, run a real model, and reach a set of explicit performance bars. That is the distance between "I have written an inference engine" and "I have written an inference engine that is usable".</p>

The instructions are in the repository at [`assignments/a3-engine-gpu/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/assignments/a3-engine-gpu). An NVIDIA GPU is required.

## The bars {#门槛}

| Item | Requirement |
| --- | --- |
| Correctness | greedy decoding matches Hugging Face transformers token by token, and every test in this book passes |
| Throughput | on the same card, model and request set, offline throughput reaches at least 60% of SGLang's |
| Latency | at concurrency 1, TPOT is no more than 1.5× SGLang's (with CUDA Graph in effect) |
| Ablations | turn off overlap scheduling, CUDA Graph, the radix cache and chunked prefill in turn, and report what each contributes |

## How to measure {#怎么测}

- Use exactly the same requests on both sides: generate the dataset with a fixed random seed (`sglang.bench_serving`'s random dataset or ShareGPT, say), submit every request at once, and measure output tokens divided by total time;
- Reuse the measurement scripts and ablation method from the [benchmarks and ablations](../perf/benchmark.md) chapter directly;
- When a bar is missed, find the reason in a profiler timeline: whether the GPU kernels are the bottleneck, whether CPU scheduling is falling behind (the gaps), and whether CUDA Graph is being hit, going through [the gap to SGLang](next-steps.md) item by item.

## What to deliver {#交付物}

A repository with the official structure, a performance report (hardware, model, request set, every number and the ablation table) and a blog post. This is exactly what "project A" in the sprint plan asks for, and every number you put on a résumé has to be one you can account for.

The other project needs no GPU: [adding the hybrid model Qwen3.5](assignment-hybrid.md), which exercises bringing up a new model and reworking the engine for "per-request state".
