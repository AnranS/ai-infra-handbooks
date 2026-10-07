# mini-sglang from Scratch

<p class="lead">mini-sglang is the SGLang team's "teaching SGLang": about 5,000 lines of core Python that still contain everything a modern inference engine needs, from radix cache, chunked prefill, overlap scheduling and tensor parallelism to FlashInfer / FlashAttention, CUDA Graph and an OpenAI-compatible online service. This handbook follows the official module layout and starts from an empty directory, one module per chapter, until you have a complete, working mini-sglang whose interfaces match the official one. Every step has tests, and the output matches Hugging Face transformers token by token.</p>

## What you will be able to do {#学完能做到}

- Explain mini-sglang's process layout: the API server, the tokenizer, the detokenizer, one scheduler process per TP rank, and every kind of message they pass over ZMQ;
- Implement the compute side yourself: an op and weight system that does not rely on `nn.Module`, streaming weight loading, a paged KV pool, the page table and the token pool, a pluggable attention backend, and the sampler;
- Implement the scheduling side yourself: continuous batching, admission control, page allocation and release, the radix cache (matching, insertion, splitting, LRU eviction, reference counting), chunked prefill and overlap scheduling;
- Explain and verify by hand the three mechanisms that make it fast: the four ways to shard a linear layer under tensor parallelism, FlashInfer's plan/run and FlashAttention's paged interface, and CUDA Graph's fixed-buffer contract;
- Read every file in the official repository, and explain the five problems this reimplementation found and fixed: why each one happens, how to reproduce it, and how to fix it.

## Learning path {#学习路线}

The chapter-by-chapter path across all the handbooks is in the [roadmap](root://roadmap/). This book belongs after "Building an inference engine from scratch" in the [Inference Systems handbook](serving://): nano_engine there builds the concepts, and here a real, complete code base at production performance turns every detail into code.

<div class="roadmap" markdown>

| Part | Chapters | Goal | Suggested time |
| --- | --- | --- | --- |
| Overview | [Architecture](overview/architecture.md) | Understand the processes, the data flow and the module layering; set up the environment | half a day |
| Computing it right | [Core data structures](compute/core.md) · [Op layer](compute/layers.md) · [Model and weights](compute/models.md) · [KV pool](compute/kvcache.md) · [Attention backend](compute/attention.md) · [Engine and sampling](compute/engine.md) | Assemble a batch by hand and run prefill + decode, matching HF token by token | 4-5 days |
| Scheduling it well | [Scheduler skeleton](schedule/scheduler.md) · [CacheManager](schedule/cache-manager.md) · [Radix cache](schedule/radix-cache.md) · [Chunked prefill](schedule/chunked-prefill.md) · [Overlap scheduling](schedule/overlap.md) | Continuous batching over many requests, prefix reuse, CPU cost hidden | 1 week |
| Serving | [Messages and ZMQ](serve/message.md) · [Tokenizer](serve/tokenizer.md) · [Scheduler IO](serve/scheduler-io.md) · [API server](serve/api-server.md) | A multi-process OpenAI-compatible service with streaming and abort on disconnect | 3-4 days |
| Faster and bigger | [Tensor parallelism](perf/tensor-parallel.md) · [GPU attention](perf/gpu-attention.md) · [CUDA Graph](perf/cuda-graph.md) · [CUDA kernels](perf/kernels.md) · [MoE](perf/moe.md) · [Benchmarks](perf/benchmark.md) | Many GPUs, GPU kernels, MoE models | 1 week |
| Wrapping up | [The gap to SGLang](wrap/next-steps.md) | Know what is still missing and pick a direction to keep going | as needed |
| Projects | [The GPU performance bar](wrap/assignment.md) · [Adding the hybrid model Qwen3.5](wrap/assignment-hybrid.md) | Reach 60% of the official build on a GPU; give the engine a linear-attention layer with per-request state | 1-2 weeks each |

</div>

## How this book works {#这本书的做法}

**Same names, same interfaces.** Our package is also called `minisgl`, and the directory layout, class names, function names and method signatures all match the official one (the official repository is pinned to commit `9a91cfa` of 2026-05-17). Each chapter opens with the files you are about to write, and the !!! upstream boxes in the text give the exact location of the matching official code, with line-numbered links, so after a chapter you can diff your version against the official one line by line.

**Every step is verifiable on a CPU.** The official implementation only supports NVIDIA GPUs. To let any machine follow along and to make every step checkable, we did three things:

1. Added a very thin device abstraction (`minisgl/utils/device.py`): real streams, events and pinned memory on CUDA, no-ops on CPU;
2. Gave every GPU op a PyTorch reference implementation: the attention backend has an extra `torch` option, and sampling, RMSNorm, RoPE and the KV-cache writes all have CPU versions;
3. Verified the GPU-only parts on a CPU through stand-ins with the same interface: FlashInfer and FlashAttention each have a PyTorch fake with the same interface, CUDA Graph has a CPU emulation (which forces every input through fixed buffers, so missing one copy makes the result wrong), CUDA kernels run on the CPU simulator from the [CUDA handbook](cuda://), and Triton kernels run in interpreter mode.

On a real GPU the same code switches to FlashInfer, FlashAttention, CUDA Graph and the custom kernels by itself.

**Every step matches Hugging Face.** Besides the unit tests, each chapter has an end-to-end check: run Qwen3-0.6B in float32 and have every greedy-decoded token match `transformers`'s `generate`. By the end the same check covers the radix cache, chunked prefill, overlap scheduling, page sizes above 1, tensor parallelism (TP=2 and TP=4), all three attention backends and CUDA Graph, across the Qwen2.5, Llama 3 (long-context RoPE) and Qwen3-MoE architectures.

!!! diff "Where we differ from the official implementation"
    Besides the CPU support above there are a few differences, each explained in a "difference from upstream" box in its chapter. Five of them fix problems in the official code, and each has a test that fails if you put the official version back: the finished flag arriving one token early under overlap scheduling, one stale message sent after EOS, a request slot reused while its batch is still on the GPU, and KV pages freed twice when a request aborts mid-prefill (chapter 11), plus a PUB/SUB broadcast that does not wait for its subscribers (chapter 12). The rest are simplifications: the model definitions are merged into one generic decoder file; tensor parallelism uses only `torch.distributed`, without the official PyNCCL built on tvm-ffi; and the fused-MoE Triton kernel is a simplified version.

## The code, and running it {#代码与运行}

All the code is in the repository's [`minisgl/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/minisgl) directory:

<!-- i18n:diagram 0ea11ca641 -->
```text
minisgl/
├── python/minisgl/      our implementation, one to one with the official python/minisgl/
├── tests/               one test file per chapter (test_ch07_scheduler.py …), plus GPU-library fakes and a CUDA self-check
├── examples/            the script behind every "what it prints" block in the text
├── tools/check.py       one command runs every check: examples, nvcc build, CPU simulator, pytest
├── tools/diagrams.py    generates the book's 15 architecture figures (SVG, following the light / dark theme)
├── tools/videos/        scene scripts and renderer for the 6 animations (edge-tts voice, frames then ffmpeg)
└── docs/                this handbook
```

```bash
cd minisgl
pip install -e ".[dev]"                                   # deps: torch, transformers, pyzmq, fastapi and friends
python -m minisgl --model Qwen/Qwen3-0.6B                 # start an OpenAI-compatible service (GPU or CPU)
python -m minisgl --model Qwen/Qwen3-0.6B --shell        # interactive chat
PYTHONPATH=python:tests pytest -q tests                   # the whole test suite (needs Qwen3-0.6B downloaded into models/)
```

The "files you will write" at the start of each chapter is what you type yourself. A good rhythm: read the idea, write your own version, then compare against this book's implementation and the official one, and finally run the chapter's tests. If you cannot answer a chapter's opening self-test, go back to the matching concept chapter in the [Inference Systems handbook](serving://).

The book has 15 architecture figures and 6 narrated, subtitled animations of 1.5 to 2 minutes each, worth watching before you read a chapter: [the life of a request](overview/architecture.md), [continuous batching and admission control](schedule/scheduler.md), [the radix cache](schedule/radix-cache.md), [overlap scheduling](schedule/overlap.md), [tensor parallelism](perf/tensor-parallel.md) and [CUDA Graph](perf/cuda-graph.md). The narration and subtitles are in Chinese.
