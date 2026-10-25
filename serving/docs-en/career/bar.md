# The interview bar: what an inference role expects you to do

<p class="lead">There is a line between "I've studied it" and "I'm ready to interview": you can explain the principles, write the code correctly under a time limit, estimate the numbers on the spot, and defend your projects under follow-up questions. This chapter breaks that line into items you can check yourself; each one says how far you need to go, where to learn it and how to verify it. The items are distilled from requirements that keep appearing in job postings for inference roles, and are grouped into the <b>baseline</b> every inference role expects, the <b>deeper items</b> for each of the three kinds of roles (inference frameworks, inference optimization, inference platforms), and the <b>frontier topics</b> that teams running inference at scale like to ask about.</p>

!!! tip "How to use this page"
    - The test is "can I do it right now", not "have I studied it". Every item is tagged with a target level: **L1 explain** (explain the principle, the trade-offs and the numbers without notes), **L2 write** (write code on your own that passes the tests), **L3 fast** (finish within a time limit, or reach a given performance threshold), **L4 deliver** (a reproducible project, report or merged PR). How each level is verified is described in the [sprint plan](root://plan/).
    - "Explain" is measured with the [mock interview scoring sheet](mock-interview.md#评分表): all four dimensions (correctness, depth, numbers, communication) must score ≥ 3.
    - Clear the baseline first, then add the deeper items for the roles you are targeting. Before applying, self-test the whole chapter; the items you fail are what's left to fix. Where a job posting asks for something that isn't here, the posting wins.

## The baseline: every inference role {#基本盘所有推理岗都要过}

### Coding {#编码}

| Requirement | The bar | Where to practise |
| --- | --- | --- |
| Algorithms | You have practised every common pattern: arrays and hashing, two pointers and sliding windows, linked lists, stacks and monotonic stacks, binary search, trees and graphs, heaps and greedy, backtracking and dynamic programming, tries and union-find. You solve ≥ 70% of medium problems correctly within 25 minutes, and can give the idea and complexity for hard ones; about 200 problems in total, with the LeetCode Top 100 done at least twice. **L3** | [Preparing for algorithm interviews](cs://algo/overview/) and the algorithm chapters |
| Model components by hand | In 30 minutes, write GQA attention with a KV cache plus RoPE; write temperature, top-k and top-p sampling and a numerically stable softmax correctly. **L3** | [Coding questions](coding.md); practice problems [GQA attention](root://practice/#/p/llm-gqa), [sampling](root://practice/#/p/llm-sampling) |
| System components by hand | In 40 minutes, write the match, insert, lock and evict operations of a Radix Cache; write a continuous-batching scheduler (token budget, chunked prefill, preemption) correctly; write a thread pool or a bounded blocking queue correctly. **L3** | Practice problems [Radix Cache](root://practice/#/p/ms-radix-cache), [scheduler](root://practice/#/p/sv-scheduler), [thread pool](root://practice/#/p/cpp-thread-pool) |
| GPU kernels by hand | In 45 minutes, write a warp reduction and a shared-memory tiled GEMM in CUDA C++, and explain what each optimization step saves. For inference platform roles, L1 is enough. **L3** | [Reduction](cuda://kernels/reduction/), [The road to an optimized GEMM](cuda://kernels/gemm/); practice problems [warp reduction](root://practice/#/p/cu-warp-reduce), [tiled GEMM](root://practice/#/p/cu-gemm-tiled) |

### Estimation {#估算}

Almost every round of an inference interview asks you to "estimate it". Estimation is the fastest way to stand out: two candidates may explain the same principle, but the one who can give the order of magnitude on the spot is far more convincing.

| Requirement | The bar | Where to practise |
| --- | --- | --- |
| Roofline | Use arithmetic intensity and the ridge point to explain why decode is memory-bound, why prefill is compute-bound, and why batching helps. **L1** | [The full journey of a token](llm://synthesis/token-journey/), [the estimation questions in the question bank](interview.md#一估算与基础) |
| Model accounting | Given any config.json, compute the parameter count, FLOPs per token and KV bytes per token (for MHA, GQA and MLA alike) within 10%. **L3** | [Estimating parameters, compute and memory](llm://inference/estimation/), [estimation practice problems](root://practice/#/?q=估算) |
| Deployment accounting | Estimate how many concurrent requests one GPU can hold, the lower bound on TPOT at batch=1, and how many GPUs a service needs; estimate the per-step communication volume of TP and EP. **L3** | [Load testing, SLOs and capacity planning](../perf/benchmark.md), [Expert parallelism and DP attention](../distributed/expert-parallel.md) |
| Training memory | Work out the per-GPU memory budget of a 70B model under ZeRO-3 and under TP + PP. **L3** | [Distributed training overview](train://basics/overview/), [ZeRO and FSDP](train://data/zero-fsdp/) |

### The inference engine {#推理引擎}

- Explain every step a request goes through from entering the service to returning tokens, and the bottleneck of each step. **L1** → [The life of a request](../engine/overview.md)
- Explain the trade-offs of continuous batching, chunked prefill and preemption (recompute or swap), what paged KV solves and what it costs, and the two ways of doing prefix caching (hashed blocks and a radix tree). **L1** → [The scheduler](../engine/scheduler.md), [Paged KV cache](../engine/paged-kv.md), [Prefix caching](../engine/prefix-cache.md)
- Explain why CUDA Graphs are used only for decode and what padding costs, what overlap scheduling saves, and why incremental detokenization keeps a context window. **L1** → [CUDA Graphs and torch.compile](../engine/graphs-compile.md), [The sampler and streaming output](../engine/sampler-api.md)
- You have written a mini engine that runs, with greedy decoding matching Hugging Face token for token. **L2** → [Building an inference engine from scratch](../engine/overview.md), [Writing mini-sglang by hand](minisgl://)
- Within 5 minutes, find the scheduler, KV allocation, CUDA Graph and model registry in the SGLang source; explain the main differences between vLLM V1 and SGLang in scheduling and prefix caching. **L3** → [SGLang source walkthrough](../source/sglang.md), [vLLM V1 source walkthrough](../source/vllm.md)

### Distributed systems and communication {#分布式与通信}

- Explain what TP, PP, EP, DP and CP each split and communicate; the communication volume and use cases of all-reduce, all-gather, reduce-scatter and all-to-all. **L1** → [Tensor parallelism](../distributed/tensor-parallel.md), [Pipeline and context parallelism](../distributed/pp-cp.md), [Collective communication](../comm/nccl.md)
- Explain the benefits, costs and KV transfer options of PD disaggregation, and why DeepEP's high-throughput and low-latency modes are designed the way they are; give the bandwidth orders of magnitude of NVLink, PCIe and RDMA networks. **L1** → [PD disaggregation and KV transfer](../distributed/pd-disagg.md), [NVSHMEM and DeepEP](../comm/nvshmem-deepep.md), [GPU interconnects and networking](../comm/interconnect.md)
- Implement DDP from scratch (gradient bucketing + overlapping communication with compute), with multi-process results matching a single process. **L2** → [Data parallelism and DDP](train://data/ddp/)

### Computer science fundamentals {#计算机基础}

Systems-leaning roles often have a whole round on fundamentals, and inference framework interviews often drill from the fundamentals down into the inference case (from virtual memory to paged KV, for example).

| Area | The bar (all **L1**) | Where to learn |
| --- | --- | --- |
| Operating systems | Processes, threads and scheduling; virtual memory, page tables and huge pages; a write from the system call down to the disk (page cache, fsync, O_DIRECT); epoll and io_uring; shared memory and CUDA IPC; namespaces and cgroups; pinned memory, DMA and NUMA; using tools like perf to find bottlenecks | [Processes, threads and scheduling](cs://os/process-thread/), [Virtual memory](cs://os/virtual-memory/), [How a write reaches the disk](cs://os/io-stack/), [epoll and io_uring](cs://os/io-models/), [Inter-process communication](cs://os/ipc/), [Containers](cs://os/containers/), [Pinned memory and NUMA](cs://os/pinned-numa/), [Performance tools](cs://os/perf-tools/) |
| Networking | TCP handshakes, congestion control and head-of-line blocking; HTTP streaming; the basic results of load balancing and queueing; one-sided RDMA operations and memory registration | [TCP](cs://net/tcp/), [HTTP and streaming output](cs://net/http-stream/), [Load balancing and queueing](cs://net/load-balance/), [The RDMA programming model](../comm/rdma.md) |
| Distributed fundamentals | Consistent hashing and sharding; replication and consensus (Raft's leader election and log replication) | [Consistent hashing and sharding](cs://dist/hash-shard/), [Replication and consensus](cs://dist/replication/) |
| Computer architecture | CPU caches and coherence; the GPU's SMs, warps, Tensor Cores and memory hierarchy, with the key numbers of mainstream cards | [A crash course in CPU architecture](cs://arch/cpu/), [The GPU's SMs and Tensor Cores](cs://arch/gpu-sm/), [The GPU memory system](cs://arch/gpu-memory/) |
| Compilers | From source to syntax tree, intermediate representations and optimization passes, loop transformations, code generation; which layer each AI compiler (torch.compile, Triton, MLIR and the like) sits at | [A crash course in compilers](cuda://framework/compiler-basics/), [The AI compiler landscape](cuda://framework/compilers/) |

### System design {#系统设计}

- The 10 questions in [System design questions](system-design.md), each finished within 45 minutes: clarify the requirements → estimate capacity → architecture → critical path → failures, scaling and observability, with numbers at every step. **L3** → [Reference answers (1)](design-answers-1.md), [Reference answers (2)](design-answers-2.md)

### Projects, open source and résumé {#项目开源与简历}

- At least one working inference-system project with comparison numbers. For example, your own mini-sglang: on the same GPU and workload its throughput reaches at least 60% of SGLang's, and you can say where every number in the performance report comes from. **L4** → [Portfolio, résumé and study plan](projects.md)
- For every project, prepare 3 hard problems, 3 numbers and 1 failed attempt, and hold up under 30 minutes of follow-up questions. **L3** → [Portfolio guide](portfolio-guide.md)
- At least 1 PR merged into an inference framework or a related open-source project; docs, tests and small reproducible bugs are all good places to start. **L4** → [Getting started with open-source contributions](portfolio-guide.md#开源贡献入门)
- A one-page résumé, with every project written as "problem → approach → numbers". **L4** → [Portfolio, résumé and study plan](projects.md)

### Mock interviews {#模拟面试}

- At least 6 full mock interviews (coding, fundamentals follow-ups, system design, project deep-dive), with all four scores ≥ 3 in the last 3. **L3** → [Scoring sheet and review](mock-interview.md), [Mock interview sets](mock-exams.md)
- Turn every topic into a card with a one-sentence answer and one follow-up question, about 150 cards in total; review the [flashcards](root://cards/) until you can answer all of them. **L1**

## Going deeper, by role {#加深项按岗位}

All three kinds of roles share the same baseline; what differs is where the interviews dig. The day-to-day work of each is described in [the role profiles](projects.md#岗位画像).

### Inference framework engineering {#推理框架开发}

- Explain engine internals down to the implementation, against the source: the scheduler, KV management and prefix caching, CUDA Graphs and overlap scheduling, speculative decoding (EAGLE, MTP), structured output. **L1** → [Advanced speculative decoding](../topics/speculative.md), [Structured output and tool calling](../topics/structured-output.md)
- How parallelism lands in a framework: TP's weight sharding and communication, EP and DP attention, KV transfer in PD disaggregation. **L2** → [Tensor parallelism from scratch](../distributed/tensor-parallel.md), [Expert parallelism and DP attention](../distributed/expert-parallel.md)
- Onboarding new models and aligning accuracy, hot weight updates, multi-LoRA serving. **L1** → [Onboarding new models and aligning accuracy](../ops/new-model.md), [Hot weight updates](../ops/weight-update.md), [Multi-LoRA serving](../ops/multi-lora.md)
- PyTorch's CUDA runtime (asynchronous execution, streams, the caching allocator), the dispatcher and custom operators; fluency in C++ or Rust besides Python. **L2** → [PyTorch's CUDA runtime](cuda://framework/cuda-runtime/), [The dispatcher and custom operators](cuda://framework/dispatcher/)
- The project that clears the bar: your own mini engine running on a GPU and compared with SGLang or vLLM under the same workload; ideally also a merged non-docs PR in an inference framework. **L4**

### Inference optimization / kernels {#推理优化--算子}

- Real-GPU performance thresholds: a reduction reaching 80% of measured bandwidth, an SGEMM reaching 70% of cuBLAS (the silver tier of the practice problems); use Nsight Compute to explain how far you are from the roofline and why. **L3** → [Profiling](cuda://tools/profiling/)
- Write a fused softmax and a tiled matmul in Triton and explain how the work splits differently from CUDA; know where tile languages one level lower, such as TileLang, fit. **L2** → [Triton](cuda://tools/triton/), [TileLang](cuda://tools/tilelang/)
- Features from Hopper on: TMA and asynchronous copies, WGMMA, CuTe's layout algebra; read the main loops of FlashMLA and DeepGEMM and reproduce the order of magnitude of their public benchmarks on a real GPU. **L3** → [Asynchronous copies and Hopper features](cuda://advanced/async-hopper/), [CuTe's layout algebra](cuda://advanced/cute-layout/), [MLA inference and FlashMLA](../moe/mla.md), [FP8 grouped GEMM](../moe/fp8-gemm.md)
- Quantization: the trade-offs of fine-grained FP8 scaling, INT4 and FP4 (MXFP4, NVFP4), GPTQ, AWQ, SmoothQuant and quantization-aware training, and estimating the gains. **L1** → [Quantization and GEMV](cuda://advanced/quantization/), [Quantized deployment in practice](../perf/quantization-deploy.md), [Low-bit inference for very large MoE](../frontier/low-bit.md)
- The gaps between kernels: PDL, megakernels, CUDA Graphs. **L1** → [PDL and megakernels](cuda://tools/pdl-megakernel/)
- The project that clears the bar: an MLA decode attention kernel or a block-scaled FP8 GEMM, with tests against a reference and a benchmark, and its fraction of peak performance in numbers. **L4** → [Portfolio guide: a GPU kernel](portfolio-guide.md#作品-cgpu-算子)

### Inference platform / deployment {#推理平台--部署}

- Kubernetes: how GPUs are scheduled and isolated, gang scheduling, operators, probes, rolling releases and autoscaling. **L1** → [How Kubernetes manages GPUs](../k8s/gpu.md), [The scheduler](../k8s/scheduling.md), [Deployment shapes](../k8s/deploy-scale.md), [Operators, quotas and troubleshooting](../k8s/operator-ops.md)
- Routing and traffic: cache-aware routing, PD pairing, load balancing, queueing and rate limiting. **L2** → [Global scheduling in disaggregated serving](../frontier/disagg-sched.md), [Load balancing and queueing](cs://net/load-balance/)
- Tiered KV and storage: offloading to CPU memory and SSD, distributed KV stores and transfer engines; which part of the path io_uring, NVMe and RDMA each handle. **L1** → [Tiered KV caching and offloading](../distributed/kv-offload.md), [KV transfer engines and distributed KV stores](../comm/kv-storage.md)
- Load testing, SLOs and capacity planning; metrics, logs and tracing. **L3** → [Load testing, SLOs and capacity planning](../perf/benchmark.md), [Production deployment and operations](../ops/deploy.md)
- Fluency in at least one of Go or Rust. **L2** → [vLLM's Rust frontend](../source/rust-frontend.md)
- The project that clears the bar: an OpenAI-compatible inference gateway with streaming, cache-aware routing, metrics and a load-test report with round-robin and least-connections baselines. **L4** → [Portfolio guide: the inference gateway](portfolio-guide.md#作品-bpd-感知的推理网关rust)

## Frontier topics teams running inference at scale ask about {#前沿专题做大规模推理的团队常问}

Roles whose postings mention "large-scale distributed inference", "MoE", "long context" or "RL systems" usually dig into the topics below on top of the baseline. The bar is **L1**: talk about each topic for 15 minutes without notes, covering the motivation, the approach and the cost, with numbers.

- Large-scale MoE inference: EP scale, DP attention, dual micro-batch overlap, hierarchical EPLB, fault tolerance and elastic scaling; a complete case study of a public large-scale MoE inference system. → [Case study: a public large-scale MoE inference system](../moe/case-study.md), [Large-scale EP deployment](../moe/ep-deploy.md), [Fault tolerance and elastic scaling](../moe/ep-elastic.md)
- New approaches to attention and decoding: MLA and FlashMLA, MTP speculative decoding, sparse attention (NSA, DSA), block-level drafts for speculative decoding. → [MLA inference](../moe/mla.md), [MTP and sparse attention](../moe/mtp-sparse.md), [New approaches to speculative decoding](../frontier/spec-next.md)
- Disaggregation and long context: KV-aware routing, xPyD, PD multiplexing, KV eviction for long context, linear attention and hybrid architectures. → [Global scheduling in disaggregated serving](../frontier/disagg-sched.md), [PD multiplexing](../frontier/pd-multiplex.md), [Long context](../topics/long-context.md), [Linear attention and hybrid architectures](../frontier/linear-attn.md)
- RL inference systems: asynchronous rollout, policy staleness, weight updates. → [RL inference systems](../frontier/rl-async.md), [Inference inside RL training](../topics/rl-rollout.md)
- New models and more hardware: structural changes in the latest open models, the platform abstraction and how non-NVIDIA accelerators are supported. → [The new generation of open models](../frontier/new-models.md), [Supporting more hardware](../ops/platforms.md)

## Self-check before you apply {#投递前自测}

Do every item below on the spot. Whatever you can't do is what's left to fix before you apply.

- [ ] Pick 3 random medium algorithm problems and solve each within 25 minutes
- [ ] Write GQA attention with a KV cache plus RoPE in 30 minutes, and a Radix Cache in 40 minutes
- [ ] Given the config.json of a model you haven't seen, compute the parameter count, KV bytes per token and the TPOT lower bound at batch=1 within 10 minutes
- [ ] Draw every step a request takes through an inference engine without notes, and explain each step's bottleneck
- [ ] Explain the communication volume and cost of TP, EP and PD disaggregation
- [ ] Explain what a file write goes through from the system call to the disk, and what a streamed response goes through from the GPU to the browser
- [ ] Pick any system design question, finish it within 45 minutes and give a capacity estimate
- [ ] For every project, name 3 hard problems, 3 numbers and 1 failed attempt
- [ ] At least 1 PR merged, and a finalized one-page résumé
- [ ] All four scores ≥ 3 in your last 3 mock interviews
