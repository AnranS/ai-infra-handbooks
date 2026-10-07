# Inference Systems

<p class="lead">This handbook is for LLM inference roles: inference framework development, inference optimization and inference platforms. The earlier handbooks answered "which language to write it in", "what the model computes" and "how GPUs compute fast"; this one answers "how an inference system puts them together". We first write a working inference engine from scratch (paged KV, continuous batching, prefix caching, a streaming API), then use it to read the vLLM and SGLang source, go deep into distributed inference, performance engineering and advanced topics, and finish with interviews, system design and a portfolio.</p>

## What you will be able to do {#学完能做到}

- Explain every step a request takes inside an inference engine, from HTTP to the last token, along with the corresponding classes and functions in vLLM and SGLang;
- Implement on your own a paged KV cache, variable-length batching, a continuous-batching scheduler (chunked prefill, preemption), prefix caching (hashed blocks and a radix tree), batched sampling and a streaming OpenAI API;
- Derive by hand how tensor, expert, pipeline and context parallelism split the work and communicate, and estimate their costs; explain the designs of PD disaggregation and hierarchical KV caching;
- Analyze communication between GPUs and between machines with the α-β model, and explain NCCL's algorithms and protocols, the RDMA programming model, DeepEP's two modes, and the design of KV transfer engines and distributed KV storage;
- Do capacity planning with open-loop load tests and simulators, locate bottlenecks with a profiler, and choose a quantization scheme for a given workload;
- Explain the inference problems of speculative decoding (including tree drafts), long context and sparse attention, structured output, multimodality and RL rollouts;
- Explain the key designs of large-scale MoE inference: MLA's two computation paths, fine-grained FP8 quantization and grouped GEMMs, hierarchical EPLB and two-batch overlap, MTP and sparse attention, and review a public production system with estimates;
- Design global scheduling for disaggregated architectures (KV-aware routing, xPyD ratios, overload control), and explain what hybrid linear-attention models, 4-bit quantization-aware training and asynchronous RL ask of inference systems;
- Handle inference interview questions, coding questions and system design with confidence.

## Learning path {#学习路线}

For the chapter-by-chapter route across all the handbooks (17 weeks, matching the sprint plan week by week, with core and optional chapters, key chapters for different directions, and dependencies across books), see the [roadmap](root://roadmap/). Below is the order within this book.

<div class="roadmap" markdown>

| Part | Chapters | Goal | Suggested time |
| --- | --- | --- | --- |
| The big picture | [The life of a request](engine/overview.md) | Build a map of the whole book | half a day |
| Building an inference engine from scratch | [Paged KV](engine/paged-kv.md) · [Variable-length batching](engine/batch-layout.md) · [Scheduler](engine/scheduler.md) · [Prefix caching](engine/prefix-cache.md) · [Sampling and API](engine/sampler-api.md) · [CUDA Graphs](engine/graphs-compile.md) | Implement every core component of an engine yourself and verify it token by token | 1.5 weeks |
| Source walkthroughs | [vLLM V1](source/vllm.md) · [SGLang](source/sglang.md) · [vLLM's Rust frontend](source/rust-frontend.md) | Map the components you wrote onto real code, and see the boundary between the serving layer and the engine | 1 week |
| Distributed inference | [Tensor parallelism](distributed/tensor-parallel.md) · [Expert parallelism](distributed/expert-parallel.md) · [Pipeline and context parallelism](distributed/pp-cp.md) · [PD disaggregation](distributed/pd-disagg.md) · [Hierarchical KV caching](distributed/kv-offload.md) | Understand how large models are deployed and what communication costs | 1 week |
| Communication and storage | [Interconnects and networks](comm/interconnect.md) · [NCCL and custom all-reduce](comm/nccl.md) · [The RDMA programming model](comm/rdma.md) · [NVSHMEM and DeepEP](comm/nvshmem-deepep.md) · [KV transfer and storage](comm/kv-storage.md) | See how data moves between GPUs and machines, and understand systems like DeepEP and Mooncake | 4–5 days |
| Performance engineering | [Load testing and capacity planning](perf/benchmark.md) · [Profiling](perf/profiling.md) · [Quantized deployment](perf/quantization-deploy.md) | Measure, find bottlenecks, choose solutions | 4–5 days |
| Advanced topics | [Speculative decoding](topics/speculative.md) · [Long context](topics/long-context.md) · [Structured output](topics/structured-output.md) · [Multimodality](topics/multimodal.md) · [Inference in RL](topics/rl-rollout.md) · [Deterministic inference](topics/deterministic.md) | Cover the frontier problems of today's inference systems | 1 week |
| Frontier: large-scale MoE inference | [MLA inference](moe/mla.md) · [FP8 and DeepGEMM](moe/fp8-gemm.md) · [Large-scale EP deployment](moe/ep-deploy.md) · [EP fault tolerance and elasticity](moe/ep-elastic.md) · [MTP and sparse attention](moe/mtp-sparse.md) · [Reviewing a public system](moe/case-study.md) | Explain every design of an inference system for DeepSeek-style models, and check public numbers with estimates | 1 week |
| Frontier: disaggregation, long context and RL inference | [Global scheduling](frontier/disagg-sched.md) · [PD multiplexing](frontier/pd-multiplex.md) · [Linear attention and hybrid architectures](frontier/linear-attn.md) · [Low bits and QAT](frontier/low-bit.md) · [Asynchronous RL and weight sync](frontier/rl-async.md) | See inference systems from the cluster's point of view: routing, ratios, new architectures, low bits and RL | 4–5 days |
| Frontier: new models and speculative decoding | [The new generation of open models](frontier/new-models.md) · [New approaches to speculative decoding](frontier/spec-next.md) | Explain what the 2026 generation of models (compressed and sparse attention, hyper-connections, new MoE) asks of engines, and load-adaptive speculative decoding | 3–4 days |
| Production and ecosystem | [Deployment and operations](ops/deploy.md) · [Hot weight updates](ops/weight-update.md) · [Choosing a framework](ops/frameworks.md) · [Multiple hardware and Ascend](ops/platforms.md) · [Multi-LoRA](ops/multi-lora.md) · [On-device inference](ops/edge.md) · [Bringing up new models and matching accuracy](ops/new-model.md) | Run the service in production, choose the right framework and form for a scenario, and get a new model running correctly on day one | 4–5 days |
| Job hunting | [Interview bank](career/interview.md) · [Coding questions](career/coding.md) · [System design](career/system-design.md) · [Portfolio and study plan](career/projects.md) · [Hardware cheat sheet](career/hardware.md) | Turn knowledge into interview performance | as needed |

</div>

Before starting this book, it is best to read [LLM Internals](llm://) (at least the Transformer anatomy and how-inference-works parts) and the first half of [Advanced CUDA](cuda://). This book cites their chapters a lot. After this book's "building an inference engine from scratch" part, we recommend continuing with [mini-sglang from Scratch](minisgl://): it guides you through a complete implementation of an inference engine following the structure of the official mini-sglang, reaching the performance of the official version, and along the way finds and fixes a few problems in the official code, which makes it a very good portfolio project.

## Threads that run through the book {#贯穿全书的主线}

- **An inference engine of your own**: starting from the LLM book's `mini_llm.py` as the model, the chapters add paged KV (`paged.py`), batched forward passes (`runner.py`), scheduling and the engine's main loop (`nano_engine.py`), prefix caching (`prefix_cache.py`, `radix.py`), and sampling and serving (`sampler.py`, `detokenizer.py`, `api_server.py`); the later distributed and topic chapters all run their experiments on it. Every step is compared token by token against "generating each request on its own".
- **Real source code**: the source walkthroughs and every chapter's "source code" boxes are based on **vLLM 0.30.0** and **SGLang 0.5.20** (September 2026). Every file path, class name, function name and command-line flag has been checked against the source.
- **Three kinds of boxes**:

!!! source "Source code"
    Where this concept is implemented in vLLM and SGLang.

!!! interview "In an interview"
    How this point is commonly asked in interviews, and how to structure the answer.

!!! inference "Inference view"
    As in the LLM book: what this knowledge means for inference performance.

## Setting up {#准备环境}

The same environment as the LLM book (CPU PyTorch, transformers, Qwen3-0.6B), plus:

```bash
uv pip install pillow                                                      # needed by the multimodal chapter
uv pip install torchvision --index-url https://download.pytorch.org/whl/cpu
```

The linear attention and multimodal chapters use Qwen3.5-0.8B (a hybrid architecture, natively multimodal, about 1.8 GB, placed in `models/Qwen3.5-0.8B`). The tensor-parallel, expert-parallel, pipeline-parallel and PD-disaggregation experiments run as multiple processes on CPU with the gloo backend of `torch.distributed`, with no GPU needed. The module files defined in the book (together with the `mini_llm.py` and other files they depend on) are packaged in [serving-code.tar.gz](assets/serving-code.tar.gz).

!!! note "How the code is verified"
    Every `python` code block and `>>>` example has actually been run under PyTorch 2.14 (CPU) and transformers 5.17. The engine's output is compared token by token against generating each request on its own; tensor parallelism, pipeline parallelism and PD disaggregation are compared against a single process; paged attention, ring attention and tree attention against standard attention; quantization, KV eviction and so on are measured by perplexity on a real model. GPU performance numbers in the book come from roofline estimates and simulators, as the text states explicitly; measure for real before going to production.
