# Ten design decisions that run through it all

<p class="lead">Having read the previous 24 chapters, three years of history compress into ten decisions: some settled in the first code release and never touched again; some made in one specific commit and validated repeatedly afterwards; and some a deliberate deferral of "do it this way for now, change it later". This chapter lists them, each with the commit where it first appears, the tests it met later and what it costs. It is the book's index, and an outline for telling an interviewer why SGLang looks the way it does.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which decisions survive from the first code release to today? Which were replaced along the way?
    2. What does each of "every TP rank schedules again and only the input is broadcast" and "scheduling and the forward pass in one process" solve, and what does each cost?
    3. Which specific commits do "borrow then give back" and "simple first, unified later" correspond to?
    4. Which decisions guarantee extensibility through "an interface plus a registry"?

??? success "Answers for the self-test (answer first, then open this)"
    1. Kept: the tree and the pool at two levels, scheduling and the forward pass in one process, the three kinds of process in a ZMQ ring, a scheduler that estimates the future rather than preempting, an attention layer that picks a backend and ignores the cache, and every rank scheduling again. Replaced: rpyc, allocation by `torch.nonzero`, 10 decode steps in a row, the copied Outlines code, its own jump-forward, the hard assumption of a page size of 1, and vLLM's layer implementations.
    2. Scheduling again reduces "every rank's batch is identical" to "identical input plus determinism" and saves broadcasting the metadata each step, at the price of CPU work times TP; one process saves an RPC hop between the scheduler and the GPU, at the price of the scheduler's CPU work standing directly in front of the GPU — which is what overlapped scheduling pays for.
    3. Borrow then give back: the first version borrowed vLLM's layers (`22085081bb`) and 26 commits from 2024-11 to 2026-01 took them back. Simple first, unified later: a page size of 1 carried the features for over a year, and #4356 of 2025-03 changed the pages once and adapted the features one by one.
    4. The attention backends (#1381, #1547), the speculative methods (`spec_registry.py`), the eviction policies (#10190), PD's transfer backends (#5328), the grammar backends (#2020), the hardware backends (`hardware_backend/`), the tool-call parsers and the routing policies (#7987).

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/principles.webp is in Chinese; put it back once the English version exists -->

## The ten decisions {#十个决定}

```bash title="principles-anchors.sh"
for h in 22085081bb d774acad5c 7d671e4ad2 cdcbde5fc3 99ec439da4 c76040e31b 2d96da813e 419a57e771 815dce0554 03464890e0; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2024-01-08  22085081bb  release initial code
2024-07-18  d774acad5c  Remove the dependency of rpyc (#646)
2024-11-19  7d671e4ad2  Enable overlap by default (#2067)
2024-07-29  cdcbde5fc3  Code structure refactor (#807)
2024-09-30  99ec439da4  Organize Attention Backends (#1547)
2025-03-12  c76040e31b  Support page size > 1 (#4356)
2024-07-19  2d96da813e  refactor model loader [unreachable code]: initial refactor (#655)
2024-11-30  419a57e771  minor: add sgl-kernel dir (#2261)
2025-01-02  815dce0554  Eagle speculative decoding part 4: Add EAGLE2 worker (#2150)
2025-01-19  03464890e0  Separate two entry points: Engine and HTTP server (#2996)
```

| # | The decision | First appearance | The tests it met | The price | Chapter |
| --- | --- | --- | --- | --- | --- |
| 1 | **The tree and the pool at two levels**: the radix tree records only slots, the pool only storage, page size 1 | the first code release `22085081bb` | the MLA pool, the tiered cache, page size > 1 and the SWA pool each only add a pool or change the key, and the tree's four core functions never changed | metadata at token granularity; page alignment had to be retrofitted | [3](../origins/radix-v1.md), [11](../perf/mla-compile.md), [16](../scale/hicache.md), [19](../scale/attention-backends.md) |
| 2 | **Estimate the future, admit conservatively, retract when wrong**, rather than admitting first and preempting | `new_token_estimation_ratio` in the first release, the dynamic coefficient and `retract_decode` in v0.2 | more conservative under DP attention (#2096), PD's decode side pre-allocating the whole stretch | the coefficient is hard to tune on a mixed workload; `--schedule-conservativeness` hands it to the user | [2](../origins/first-commit.md), [9](../service/v02.md), [17](../scale/pd.md) |
| 3 | **Scheduling and the forward pass in one process**, with tokenizing and detokenizing each its own | the first release; settled after #646 removed rpyc | overlapped scheduling moved the forward pass to a thread rather than a process; the gRPC entry still starts the scheduler through the Engine | the CPU's scheduling stands directly in front of the GPU, so it has to be overlapped | [2](../origins/first-commit.md), [6](../service/processes.md), [12](../perf/overlap.md) |
| 4 | **Every TP rank schedules again and only the input is broadcast** | #646 (2024-07-18) | under DP attention the ranks schedule different requests but synchronise the shapes each step; speculative decoding and PD both rest on it | scheduling CPU x TP; pickling a large request (multimodal) is slow | [6](../service/processes.md), [13](../perf/multi-gpu.md), [15](../scale/eagle.md) |
| 5 | **Directories by lifetime, the batch in three layers, the attention layer only picks a backend** | #807 (2024-07-29), #1543 / #1547 (2024-09-30) | a new backend only adds a file; overlapped scheduling changed two classes; the pool's implementation can be swapped | many concepts (three batches, two pools, one backend) | [10](../perf/restructure.md) |
| 6 | **Hide the CPU behind the GPU**: CUDA graphs, overlapped scheduling, future tokens | #612 (2024-07-13), #2067 (2024-11-19) | every new feature (retraction, constrained decoding, mixed batches, speculation, PD) has to be reconciled with the overlap afresh | races from concurrency; everywhere that reads a token has to handle a placeholder | [9](../service/v02.md), [12](../perf/overlap.md) |
| 7 | **An interface plus a registry for extensibility** | the attention backends #1381 / #1547; the grammar backends #2020; generalised afterwards | the backends went from 2 to over 30, a dozen speculative methods, and the same for the eviction policies, transfer backends, hardware backends, parsers and routing policies | the interface grows with the features to over 20 methods; a newcomer has to learn the registries first | [4](../origins/fsm-jump.md), [10](../perf/restructure.md), [15](../scale/eagle.md), [19](../scale/attention-backends.md) |
| 8 | **Borrow then give back**: the model layers from vLLM, the kernels from FlashInfer, then owned step by step (sgl-kernel) | borrowed in the first release; sgl-kernel created #2261 (2024-11-30); 26 remove-vllm commits | only with its own kernels could it do the per-token quantization speedup, several kinds of hardware and its own upgrade pace | a year of compat commits; a version matrix | [8](../service/borrow-vllm.md), [14](../perf/sgl-kernel.md) |
| 9 | **Simple first, unified later**: the page size, a single entry file, a single OpenAI adapter, its own jump-forward | each in its own first version; #4356 (2025-03-12), #2996, #7167 and #4032 are the moments of unification | each unification has to adapt every existing feature at once | the technical debt is repaid all at once | [4](../origins/fsm-jump.md), [19](../scale/attention-backends.md), [20](../platform/entrypoints.md) |
| 10 | **A new feature becomes a worker, a mixin or a separate runtime, and does not touch the scheduler's trunk** | the EAGLE worker #2150; PD's mixins #4654; `multimodal_gen/` #12484 | the scheduler only has to understand "several tokens per step" and "two loops"; diffusion got its own runtime | the worker gets thick, the mixins swell `Scheduler`'s method count, the repository grows | [15](../scale/eagle.md), [17](../scale/pd.md), [23](../platform/multimodal-diffusion.md) |

![Figure: where the ten decisions sit on the timeline](../assets/figures/sgl-ten-decisions.svg){.aig-svg}

## How they relate {#它们之间的关系}

The ten are not parallel; several depend on others:

- **1 → 11, 16, 19**: the tree and the pool at two levels is the precondition for the MLA pool, the tiered cache and the later page-size change each touching only one layer.
- **3 + 4 → 6**: scheduling and the forward pass in one process, and every rank scheduling again, mean that "hide the CPU behind the GPU" can only be done with thread-level overlap and future tokens, not with another scheduling process.
- **5 → 7, 10**: only once the directories and the batch were layered was there anywhere to put an interface plus a registry, or a worker or mixin.
- **8 → 14, 18**: only after taking ownership of the kernels was there a home for the DeepGEMM, permute and fused quantization operators that large-scale EP needs.
- **9 is the other face of 1 through 8**: every "do it this way for now" decision was unified at some moment, and the unifying commits are the ones most worth reading in this book.

## The decisions that were replaced {#被换掉的决定}

What did not survive is worth remembering too: rpyc (half a year), allocation by `torch.nonzero` (nine months), 10 decode steps in a row (ten months), the copied Outlines code (a month), its own jump-forward (fourteen months), vLLM's layer implementations (one to two years), `VerlEngine` (three and a half months) and `HiRadixCache` as a separate class (nineteen months, merged into the unified tree). What they have in common: each was the shortest path to making a feature work at the time, and each had something more general take over when it was replaced. Whether a decision was good is judged not by how long it lived but by whether it dragged anything else down when it was replaced — and none of these did.

## Exercises {#练习}

**1. Find one of your own.** From [chapter 24](../platform/codebase-2026.md)'s directory map, pick a directory this book has no chapter on (`lora/`, `sampling/` or, beyond `constrained/`, `parser/`), use [the next chapter](archaeology.md)'s method to find its first commit and its largest restructuring, and judge which of the decisions above it follows.

??? success "A way to approach it"
    `git log --reverse --date=short --format='%ad %h %s' "$REF" -- python/sglang/srt/lora | head`, then look for the commit with the largest `--stat`; LoRA's evolution (started 2024-09, with two restructurings of `LoRAManager` in 2025, #6994 and #7412) follows decision 7 (made into a backend) and decision 10 (does not touch the trunk).

**2. A counterexample.** Find a commit that conflicts with the ten (a large stretch of feature-specific logic added straight into `scheduler.py`, say) and see whether it was split out later.

??? success "A way to approach it"
    `git log --stat --format='%ad %h %s' "$REF" -- python/sglang/srt/managers/scheduler.py | grep -B3 'scheduler.py.*+[0-9]\{3\}'` finds a commit adding hundreds of lines at once, and then look for the same logic in `scheduler_components/` and the mixins.

!!! interview "How to explain it"
    To explain how you understand SGLang's architecture, do not start from the directory tree. Pick three decisions and explain the why (1, 4 and 6 are a good set: the tree and the pool at two levels, every rank scheduling again, hiding the CPU behind the GPU), give a commit id and a price for each, and then use one replaced decision to show you know it evolved. Ten minutes of that shows more judgement than reciting how the modules relate.

## Summary {#小结}

- [x] Ten decisions: a two-level cache, admission by estimate, one process, scheduling repeated per rank, layered directories, hiding the CPU, interfaces and registries, borrow then give back, simple first and unified later, and new features that do not touch the trunk.
- [x] They depend on each other: the two-level cache holds up every later pool; one process plus repeated scheduling decided how the overlap had to be done; only after the layering were there registries and workers.
- [x] The replaced decisions are valuable too: each was the shortest path at the time and dragged nothing else down when it went.
