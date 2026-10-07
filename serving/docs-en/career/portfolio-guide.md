# A guide to portfolio projects: design documents, milestones and open-source contributions

<p class="lead">The <a href="../projects/">portfolio, résumé and study plan</a> chapter listed project directions; this chapter breaks the three most distinguishing ones (a PD-aware inference gateway, a GPU operator, and an inference performance visualization tool) into executable plans: what kind of design document to write first, how many milestones to split into, and what numbers accept each milestone. It ends with a starter route for open-source contributions: how to read the code of a few major projects, how to pick a first issue, and how to write a PR that gets merged easily.</p>

## Write the design document first {#先写设计文档}

Spend half a day on a one-page design document before writing code, and it becomes the outline you present the project from in interviews. A suggested structure:

| Section | What to write | Example (the gateway) |
| --- | --- | --- |
| Background and goals | what problem it solves, with quantifiable goals | "in a multi-instance deployment, cache-aware routing lowers P99 TTFT by 50% versus round-robin" |
| Non-goals | state explicitly what you won't do, to bound the scope | no authentication, no multi-model routing |
| Design | an architecture diagram, key data structures, interfaces, a request's full path | an approximate prefix tree, the instance health state machine, the PD pairing flow |
| Trade-offs | at least two "it could be this way or that way" decisions, with reasons | keeping an approximate index in the router's memory vs subscribing to instances' KV events |
| Validation plan | how correctness is tested, how performance is tested, what the control group is | round-robin and least-connections as two baselines; three load levels |
| Milestones | each step's deliverable and acceptance numbers | see below |
| Risks | the likeliest failure points and fallbacks | the real engine's PD interface changing → target one fixed version first |

Have someone read it afterwards; if they can restate the design without reading the code, it is clear enough.

## Project B: a PD-aware inference gateway (Rust) {#作品-bpd-感知的推理网关rust}

The goal: a gateway in front of several inference instances (SGLang or vLLM) supporting an OpenAI-compatible interface, cache-aware routing, instance pairing for PD disaggregation, plus the health checks, retries and metrics production needs. Related chapters: [prefix caching](../engine/prefix-cache.md), [PD disaggregation](../distributed/pd-disagg.md), [global scheduling in disaggregated architectures](../frontier/disagg-sched.md).

| Milestone | Deliverable | Acceptance |
| --- | --- | --- |
| M1: proxy | an OpenAI-compatible `/v1/chat/completions`, SSE streaming passthrough, request logs | streaming output byte-identical to connecting to an instance directly; the gateway's own added latency at P99 < 1 ms |
| M2: routing | three policies: round-robin, least-connections and cache-aware (an approximate prefix tree per instance), falling back to least-connections when the load gap exceeds a threshold | numbers for cache hit rate and TTFT improvement over round-robin on a multi-turn workload; a sensitivity curve for the threshold |
| M3: PD and reliability | prefill / decode instance pairing (against SGLang's PD mode); health checks, circuit breaking, retries, graceful shutdown | fault injection: killing an instance moves requests away automatically, with numbers for failure rate and recovery time |
| M4: observability and load testing | Prometheus metrics (per-instance queue, hit rate, TTFT / TPOT percentiles); load-test scripts and a report | a comparison table of three load levels × three policies; a README that lets a stranger run it in 10 minutes |

Bonus: unify the cost as "predicted TTFT" and reject early under overload (the simulation in [global scheduling](../frontier/disagg-sched.md#路由缓存与排队的权衡) can serve directly as the design basis); contribute the idea or the code to SGLang's Rust gateway.

## Project C: a GPU operator {#作品-cgpu-算子}

The goal: an operator that shows you "can get down to the hardware", choosing one of two: an MLA decode kernel (following FlashMLA's ideas, Triton first then CUDA), or a block-scaled FP8 GEMM (following DeepGEMM). Related chapters: [MLA inference](../moe/mla.md), [FP8 and grouped GEMM](../moe/fp8-gemm.md), and the CUDA handbook's [Hopper asynchronous programming](cuda://advanced/async-hopper/) and [CUDA portfolio](cuda://career/projects/).

| Milestone | Deliverable | Acceptance |
| --- | --- | --- |
| M1: correct | a naive implementation + tests comparing against a PyTorch reference (many shapes, edge cases) | the maximum error within an agreed range (relative error for the FP8 GEMM, bf16 tolerances for MLA) |
| M2: fast | tiling, shared memory, Tensor Cores (WGMMA / mma), pipelining | report "the share of peak bandwidth" for memory-bound configurations and "the share of peak TFLOPS" for compute-bound ones |
| M3: explain | a Nsight Compute report: where the remaining gap is (memory access, pipeline bubbles, bank conflicts, occupancy) | at least one record of "made a change based on ncu's conclusion and measured the gain" |
| M4: integrate | plug it into PyTorch as a custom operator (or replace the corresponding kernel in mini-sglang directly) | numbers for the change in end-to-end decode latency or prefill throughput |

What is most convincing in an interview is not "it reached so many TFLOPS" but "I know where the remaining 30% goes, and why I stopped optimizing".

## Project D: an inference performance visualization tool (optional) {#作品-d推理性能可视化工具可选}

The goal: turn PyTorch profiler traces and inference engine logs into one interactive timeline: each step's batch composition, KV usage, CPU and GPU idle time, and a breakdown of each request's TTFT. Related chapter: [profiling inference engines](../perf/profiling.md).

| Milestone | Deliverable | Acceptance |
| --- | --- | --- |
| M1: parsing | read profiler output in Chrome trace format and engine logs, aligning the timelines | works on output from both SGLang and vLLM |
| M2: views | the timeline, per-step batch and KV usage, a per-request TTFT breakdown | an online demo |
| M3: diagnosis | automatically spot common problems: too much CPU scheduling, CUDA Graph misses, prefill interrupting decode, communication not overlapped | use it to locate a real problem and write it up as a case study |

## Getting started with open-source contributions {#开源贡献入门}

**Entry points for reading code.** Read this handbook's [vLLM walkthrough](../source/vllm.md) and [SGLang walkthrough](../source/sglang.md) first, then enter each project here:

| Project | Where to start reading | A good first kind of contribution |
| --- | --- | --- |
| SGLang | `python/sglang/srt/managers/` (the scheduler and TokenizerManager), `srt/model_executor/`, `srt/layers/`; the Rust gateway is a separate subproject | support for new models, docs and examples, reproducible bugs, small gateway features |
| vLLM | `vllm/v1/core/` (scheduling and KV management), `vllm/v1/worker/`, `vllm/model_executor/`, `csrc/` | filling test gaps, docs, model support, locating performance regressions |
| FlashInfer | `include/flashinfer/` (kernels as CUDA headers), `python/flashinfer/` | kernel tests and benchmarks, support for new shapes |
| Mooncake | `mooncake-transfer-engine/`, `mooncake-store/` | tests for the transfer engine, integration with inference frameworks |

**Picking an issue.** Start with those labeled good first issue / help wanted; better still are problems you found while using the project, since you can already reproduce them, which is the biggest advantage for getting merged. Comment on the issue first describing how you plan to do it, to avoid duplicating someone else's work.

**Writing a PR.** Make the title say clearly what changed; in the description write the motivation, reproduction steps, the approach, and how it was tested, and for performance changes always attach before-and-after data and the measurement method; the smaller the PR, the easier it merges, so split large changes into several. Before submitting, run the project's format checks (usually pre-commit) and the relevant tests; some projects require signed commits (vLLM, for example, requires `Signed-off-by` on every commit). Respond to review comments promptly, and when you disagree, bring reasons and data.

**Pace.** The sprint plan targets 3–5 PRs, at least 1–2 of them substantive non-documentation changes. The path: docs and tests → reproduce and fix a bug → a small feature. Every merged PR becomes a complete story in an interview: finding the problem, locating it, fixing it, verifying it.

!!! interview "In an interview"
    Present a project in the order "problem → approach → numbers → trade-offs", and prepare one "failed attempt" per project: what you tried, why it didn't work, what you learned. When the interviewer digs into details, the "trade-offs" section of your design document is where your answers come from.

## Summary {#小结}

- [x] Write a one-page design document first: goals, non-goals, design, trade-offs, validation plan, milestones, risks.
- [x] Projects B (the gateway), C (a GPU operator) and D (a visualization tool) each split into about four milestones, every one with quantifiable acceptance.
- [x] Start open-source contributions from entry points for reading code and from problems you can reproduce yourself; keep PRs small and complete, with reproduction steps, tests and before-and-after data.
