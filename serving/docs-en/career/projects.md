# Portfolio, résumé and study plan

<p class="lead">Inference roles value "having really done it": one project with numbers, comparisons and clearly explained trade-offs beats a long list of familiar terms. This chapter gives several project directions, from shallow to deep, that fit on a résumé (most can be built directly on this handbook's code), how to describe them on a résumé, and a study plan that strings the four handbooks together.</p>

## Role profiles {#岗位画像}

Inference-related roles fall roughly into three kinds, with different emphases:

| Direction | Day-to-day work | What matters most |
| --- | --- | --- |
| Inference framework engineering | developing scheduling, KV management, parallelism and new model support on vLLM/SGLang or an in-house engine | understanding engine architecture, Python/C++ engineering skills, the ability to read large codebases |
| Inference optimization / kernels | writing and tuning kernels (attention, GEMM, MoE, quantization), profiling and performance analysis | CUDA/Triton, roofline analysis, hardware knowledge |
| Inference platform / deployment | large-scale deployment, routing, elastic scaling, cost optimization, SLO guarantees | system design, capacity planning, load testing and monitoring, distributed systems |

Interviews for all three cover the fundamentals in [the interview question bank](interview.md); what differs is where they dig deeper.

## Project directions {#项目方向}

### 1. A GPU version of the mini inference engine (framework direction, ★ recommended) {#1-迷你推理引擎的-gpu-版本框架方向-推荐}

One ready-made path is to follow [writing mini-sglang by hand](minisgl://) chapter by chapter: its structure and interfaces match the official mini-sglang, it uses FlashInfer, FlashAttention and CUDA Graphs directly on GPUs, it already covers everything listed below, and you can run comparison experiments on top of it.

Building on this handbook's mini engine:

- Replace the reference attention with FlashInfer's or FlashAttention's paged/varlen interfaces, with one kernel for the whole batch;
- Add CUDA Graphs (captured in buckets by batch size) and asynchronous scheduling;
- Implement tensor parallelism with NCCL;
- Implement an OpenAI-compatible service and load test it;

**Deliverables**: a code repository, a comparison with vLLM on the same model and workload (the gaps in throughput, TTFT and TPOT with an analysis of why), and a technical blog post. Being able to say "my engine is 30% slower than vLLM, and the time goes to CPU overhead and kernel X" is far more credible than claiming "faster than vLLM".

### 2. Fused operators and kernel optimization (optimization direction) {#2-融合算子与-kernel-优化优化方向}

Implement key inference operators in Triton or CUDA, each with correctness tests and a roofline analysis:

- fused add + RMSNorm, SiLU × up (optionally with FP8 quantized output), RoPE;
- paged decode attention (supporting GQA, optionally split-KV);
- W4A16 GEMV / GEMM;

See the CUDA handbook's [portfolio projects](cuda://career/projects/) and its per-operator chapters. Plugging your own kernel into the mini engine or vLLM and measuring the end-to-end gain makes a complete story.

### 3. Quantization experiments (optimization direction) {#3-量化实验优化方向}

Systematically compare quantization schemes on an open model: RTN, GPTQ, AWQ, SmoothQuant, FP8 (at different scaling granularities), KV Cache quantization, evaluating accuracy with lm-evaluation-harness and measuring throughput and latency with vLLM; analyze "which layers are sensitive and why". This handbook's [quantization in deployment](../perf/quantization-deploy.md) chapter is the starting point.

### 4. A service tuning report (platform direction) {#4-服务调优报告平台方向}

Pick a model, deploy it on both vLLM and SGLang, plot latency-throughput curves with open-loop load tests, adjust the key parameters one by one (token budget, concurrency cap, prefix caching, CUDA Graphs, quantization, speculative decoding), record each parameter's effect on goodput, and finally produce a report saying "under SLO X, one GPU sustains Y req/s, and the optimal configuration is Z". Reports like this map directly onto the day-to-day work of platform roles.

### 5. Prototypes of system topics (advanced) {#5-系统专题原型进阶}

- **A cache-aware router**: implement a router in front of several vLLM instances, routing by KV events or an approximate prefix tree, and compare hit rate and TTFT against random routing;
- **A PD disaggregation experiment**: build a minimal 1P1D system with vLLM's KV connector, and measure the KV transfer overhead and the improvement in TPOT jitter;
- **Long-tail optimization of RL rollout**: implement partial rollout and measure the gain in utilization (see [inference in RL training](../topics/rl-rollout.md)).

### 6. Open-source contributions {#6-开源贡献}

Start with good first issues, documentation, tests and model support, and work up to bug fixes and performance improvements. A few suggestions:

- Read and follow the project's contribution guide first. vLLM, for example, explicitly requires: check whether the same work already exists before opening a PR; isolated trivial changes are not accepted; AI-assisted contributions must be understood by the submitter, who must be able to defend every line, and the PR description must state how AI was used along with the test commands and results;
- Fixing one real bug is worth more than ten typo corrections;
- The design trade-offs you learn in issue and PR discussions are themselves good interview material.

## How to write the résumé {#简历怎么写}

- **Write each project as "what you did + the resulting numbers + the key techniques"**: for example, "implemented an inference engine with paged KV and continuous batching, cutting the total time for 6 concurrent requests from 4.7 s of one-by-one generation to 1.8 s; profiling located per-request attention taking 82% of the time, and switching to a batched kernel lowered per-step latency by X%";
- **Numbers need a baseline**: improved relative to what? On what hardware, model and workload? How far from the theoretical floor?
- **Write less "familiar with" and more "have done"**: list the source modules you have read, the components you have implemented, the problems you have solved;
- **Be ready for follow-ups**: for every number on your résumé, the interviewer may ask "how did you measure it", "why that number", "where was the bottleneck".

## The condensed study plan (about 12 weeks) {#精简版学习计划约-12-周}

The full plan is 17 weeks: the week-by-week study, practice, output and acceptance are in [the sprint plan](root://plan/), and a chapter-by-chapter version (marking each chapter as required, recommended or optional, highlighting priorities for the framework, optimization and platform directions, with progress tracking) is in [the study roadmap](root://roadmap/). When time is tighter, compress to about 12 weeks per the table below (dropping two frontier topics and project C, with a shorter interview preparation).

| Week | Content | Output |
| --- | --- | --- |
| 1 | the chapters of [Advanced Python](python://) you are not comfortable with (concurrency, profiling, testing) | able to write and test engineering code fluently |
| 2–4 | [LLM Internals](llm://): fundamentals and math → Transformer anatomy → how inference works | hand-write `mini_llm`, able to do every estimation problem |
| 5–7 | [Advanced CUDA](cuda://): GPU fundamentals → classic operators → Tensor Cores, FlashAttention, quantized GEMV and the toolchain | write several working, tested kernels |
| 8 | this handbook: building an inference engine from scratch | the mini engine running |
| 9 | this handbook: source code walkthroughs | able to draw vLLM's/SGLang's request paths |
| 10 | this handbook: distributed inference | able to derive TP/EP by hand and explain PD disaggregation's trade-offs |
| 11 | this handbook: performance engineering and advanced topics | able to do capacity planning; able to explain speculative decoding, long context and structured output |
| 9–12 | in parallel with study: do one project (one of directions 1–4 above) | a repository + a report/blog post |
| 12 | interview preparation: question banks, implementation problems, system design, mock interviews | able to present the project and the bank's frequent questions fluently |

This is a job-hunting sprint pace (15–20 hours a week, reading closely only the required chapters). The suggested durations on each handbook's home page assume close reading from zero, about 7–8 months in total; with more time, follow that pace, and for the inference optimization direction especially, the CUDA part is worth several more weeks to actually write the kernels.

On rhythm: run some code or read some source every day rather than only reading prose; after each chapter, check yourself with the self-test at its start and reread what you can't answer; write a short study note every week, and over time it becomes blog posts and interview material.

## Summary {#小结}

- [x] Inference roles split into framework, optimization and platform directions, with the same fundamentals and different depth.
- [x] The most valuable projects are those with "numbers, comparisons and a clearly explained bottleneck": a GPU version of the mini engine, fused operators, quantization experiments, a service tuning report, system prototypes, open-source contributions.
- [x] Every number on a résumé needs a baseline and must hold up under follow-ups.
- [x] In the order "Python → LLM internals → CUDA → inference systems → a project → interviews", about 12 weeks covers it systematically.
