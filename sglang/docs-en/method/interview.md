# How to talk about SGLang in an interview

<p class="lead">The commonest failure in talking about an open-source project in an interview is turning it into a feature list: RadixAttention, zero-overhead scheduling, PD disaggregation, EP… every word correct, and nothing to distinguish you from someone who read the official blog. This book's material lets you tell it differently: chronologically, as "the problem → the first version → the trade-off → what changed afterwards", with every point landing on a commit id and a price. This chapter gives a template for telling it, outlines for twelve frequent questions (each pointing back to a chapter), and a few things easy to get wrong.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Describing SGLang's architecture in three minutes, which three points would you pick?
    2. How do you answer "where is SGLang faster than vLLM" without sounding like the blog?
    3. Pressed on "what overlapped scheduling costs", which specific commits can you name?
    4. Which statements are out of date and should not be said in an interview any more?

??? success "Answers for the self-test (answer first, then open this)"
    1. A good set: the first code release's skeleton (three kinds of process, a two-level KV pool, admission by estimate) → the performance engineering of H2 2024 (CUDA graphs by default, overlapped scheduling and future tokens, the layered directories) → the scaling of 2025 (PD disaggregation as the precondition for large-scale EP, with workers and mixins leaving the trunk alone). One commit id and one price per point.
    2. By phase and with conditions: v0.2's 3.1 times came from CUDA graphs by default, dynamic admission with retraction and decoding in a run (small batches, Llama-70B, vLLM 0.5.2); v0.4's 1.1 times from overlapped scheduling; and the gains in the DeepSeek case from DP attention plus EP plus PD plus TBO plus EPLB. Then add that the two have absorbed each other since and the difference is mostly in engineering details.
    3. Races (#1712), illegal CUDA memory accesses (#2048, #2070), retraction (#1860), constrained decoding (#2095, #2377), mixed batches (#2158), multimodal disabled for a while (#2235) — a month from the first commit to becoming the default.
    4. "SGLang is a language" (the frontend is an optional component now), "structured output is accelerated with jump-forward" (deleted in 2025-03, the feature is in the grammar libraries), "one token per page" (the page size is configurable), "it depends on vLLM's layers" (taken back), "it schedules through rpyc" (removed in 2024-07).

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/interview.webp is in Chinese; put it back once the English version exists -->

## A template for telling it {#讲法的模板}

Four sentences per point: **the problem at the time → how the first version did it → what the trade-off was → what changed afterwards**. Overlapped scheduling, for instance:

> Before October 2024 the scheduler waited for the GPU at every step before forming the next batch, leaving gaps on the GPU at small batches (the problem). From #1687 the forward pass moved to another thread and the scheduler ran one step ahead, with the tokens not yet sampled standing in as negative numbers in the next batch and the forward thread resolving them on the GPU (the first version). The price was that every piece of logic assuming "the result is known" had to change, and a month went into fixing the races in retraction, constrained decoding and mixed batches before it became the default (the trade-off). The v0.4 blog's number is 1.1 times; and in 2025 speculative decoding's v2 brought drafting and verification into the same overlap (what changed afterwards).

Those four sentences have a date, a commit id, a number and a price, and no adjectives. Prepare five or six points like that and pick three by where the interviewer is heading.

## Twelve frequent questions {#十二个高频问题}

| The question | The outline | Chapter |
| --- | --- | --- |
| How do SGLang's and vLLM's design starting points differ? | different starting points (LM programs vs one request's memory efficiency), similar destinations; the first version had a radix tree, per-token paging and regex constraints on day one | [1](../origins/paper.md) |
| How is RadixAttention implemented? | a tree node holds a stretch of tokens and the slots; matching and insertion split in the middle of an edge; lock on admission, insert and deduplicate on completion; leaf LRU; the matching bug fixed 8 days after the release | [3](../origins/radix-v1.md) |
| How is structured output accelerated? | an FSM mask; the compressed FSM and jump-forward reduced to a new request; deleted in 2025-03 and handed to xgrammar | [4](../origins/fsm-jump.md) |
| How does multi-card scheduling stay consistent? | every rank schedules again and only the input is broadcast; DP attention synchronises the shapes each step with IDLE batches keeping step | [6](../service/processes.md), [13](../perf/multi-gpu.md) |
| Why is SGLang fast? | by version: v0.2's three sources; v0.4's overlap; the DeepSeek case's five pieces; state the conditions | [9](../service/v02.md), [12](../perf/overlap.md), [18](../scale/large-ep.md) |
| How should an inference engine's modules be divided? | directories by lifetime, the batch in three layers, attention only picking a backend; and the three things that bought | [10](../perf/restructure.md) |
| How is MLA computed and stored at inference? | a latent vector plus the RoPE dimensions; weight absorption; a second pool with the tree unchanged | [11](../perf/mla-compile.md) |
| How is speculative decoding integrated? | as a worker; slots allocated first and rolled back after; one verification with a tree mask; the reconciliations with retraction, DP and the page size | [15](../scale/eagle.md) |
| How do you make a KV cache multi-tier? | a node with a host_value; the controller's two threads moving layer by layer; three write policies; a storage tier keyed by a content hash | [16](../scale/hicache.md) |
| How is PD disaggregation implemented? | decode handshakes and pre-allocates; prefill writes one-sidedly by chunk; an 81-line transfer interface; the three motives | [17](../scale/pd.md) |
| How is DeepSeek-V3 deployed at high throughput? | PD plus two DeepEP modes plus two GEMM layouts plus TBO plus EPLB; the blog's numbers and where they come from | [18](../scale/large-ep.md) |
| How does an inference engine support RL? | three ways to swap weights, an address-preserving release and resume, SPMD embedding; where the interfaces came from | [22](../platform/rl.md) |

Every row's outline expands into the four sentences above; the "how to explain it" boxes in the chapters are the longer versions.

## Things easy to get wrong {#容易说错的地方}

- **Treating the frontend as the present.** "SGLang is a language" is the definition of early 2024; today's README says it is an inference framework. Mentioning the frontend is fine, as long as you say it is an optional component now ([chapter five](../origins/frontend.md)).
- **Treating the blog's numbers as universal.** The 3.1 times has conditions of model, GPU, baseline version and batch range; give the conditions with the number.
- **Treating jump-forward as a current feature.** #4032 of 2025-03-03 deleted the implementation from the scheduler, and xgrammar does the same thing inside the library.
- **Calling "every rank schedules again" a waste.** It is deliberate: CPU traded for consistency, saving the metadata broadcast each step.
- **Calling PD disaggregation only a latency measure.** Two of the blog's three motives are about DeepEP and DP attention — it is the precondition for large-scale EP.
- **Calling sgl-kernel "all our own kernels".** The main force in attention and GEMM comes from external libraries, and what is written in-house is the fused operators and the glue.
- **Quoting out-of-date paths.** `srt/managers/router/`, `controller/`, `hiradix_cache.py` and the top-level `sgl-kernel/` are all gone; state the year when you mention a historical path.

## A one-page timeline {#一页纸的时间线}

Worth going through before an interview:

```bash title="one-page-timeline.sh"
for h in 22085081bb 01ca82d765 26f0bedc8f 0463f7fb52 d774acad5c 665815969a cdcbde5fc3 e1eae1fd15 f86c1e611f 99ec439da4 dbec2f1847 7d671e4ad2 976bc302e5 cbedd1db1d 419a57e771 815dce0554 6c7a152c5a c76040e31b c7c7dbebbe f44db16c8e 0d47788025 70c471a868 ce32bc2ba9 53ca15529a 7bc1dae095 b36afed4a7 49dfa1d891; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-80
done | sort
```

```text title="output"
2024-01-08  22085081bb  release initial code
2024-01-16  01ca82d765  fix radix cache match (#7)
2024-02-05  26f0bedc8f  jump-forward rename (#144)
2024-05-27  0463f7fb52  Support data parallelism (static) (#480)
2024-07-13  665815969a  Enable cuda graph by default (#612)
2024-07-18  d774acad5c  Remove the dependency of rpyc (#646)
2024-07-29  cdcbde5fc3  Code structure refactor (#807)
2024-08-05  e1eae1fd15  Support MLA for DeepSeek-V2 with Triton - step 1 (#905)
2024-09-29  f86c1e611f  Move scheduler code from tp_worker.py to scheduler.py (#
2024-09-30  99ec439da4  Organize Attention Backends (#1547)
2024-10-16  dbec2f1847  Launch a thread to overlap CPU and GPU (#1687)
2024-11-16  976bc302e5  Support DP MLA (#1970)
2024-11-19  7d671e4ad2  Enable overlap by default (#2067)
2024-11-23  cbedd1db1d  [router] cache-aware load-balancing router v1 (#2114)
2024-11-30  419a57e771  minor: add sgl-kernel dir (#2261)
2025-01-02  815dce0554  Eagle speculative decoding part 4: Add EAGLE2 worker (#2
2025-02-23  6c7a152c5a  Hierarchical Caching for SGLang (#2693)
2025-03-12  c76040e31b  Support page size > 1 (#4356)
2025-03-19  f44db16c8e  [Feature] Integrate DeepEP into SGLang (#4232)
2025-03-21  c7c7dbebbe  [PD] Release initial code (#4654)
2025-05-25  0d47788025  Support overlapping two batches (#4068)
2025-06-16  70c471a868  [Refactor] OAI Server components (#7167)
2025-07-26  ce32bc2ba9  Extract update_weights from RL Engine to SGLang to keep 
2025-09-11  53ca15529a  Implement Standalone gRPC Server for SGLang Python Sched
2025-10-10  b36afed4a7  Separate allocation logic from scheduler (#11313)
2025-11-06  7bc1dae095  WIP: initial multimodal-gen support (#12484)
2025-12-05  49dfa1d891  [model-gateway] change sgl-router to sgl-model-gateway (
```

27 commits in order, one for each of this book's 27 chapters, the one most worth remembering. Remember the date, the problem it solved and its price, and that is enough to keep the story going under any follow-up.

## Exercises {#练习}

**1. Record yourself.** Pick three points and talk for a minute on each as "the problem → the first version → the trade-off → afterwards", then listen back: are there sentences with no date and no price?

**2. Prepare in reverse.** Write one likely follow-up for each of the twelve questions, and find with this book's scripts the command that answers it.

??? success "A way to approach it"
    For "why does overlapped scheduling stand in with negative numbers rather than waiting for the result", `resolve_future_token_ids` in `git show b48edff67f` answers it; for "why does PD's decode pre-allocate the whole stretch", v0.4.6's `_pre_alloc`.

**3. Update.** Before the interview, set `REF` to the latest `main` and run [the previous chapter](archaeology.md)'s `keyword-first.sh` to see whether a new concept is worth adding to your timeline.

!!! interview "How to explain it"
    One last piece of advice: state your sources unprompted. "This is what I pieced together from the repository's commit history, with the baseline at main in October 2026" — it tells the interviewer your conclusions are checkable, and gives them a direction: when they press for detail, you have a commit id to answer with.

## Summary {#小结}

- [x] The way to tell it: the problem → the first version → the trade-off → afterwards, each point with a date, a commit id, a number and a price.
- [x] Twelve frequent questions each have an outline and a chapter; a few statements are out of date and should be dropped.
- [x] A one-page timeline of 27 commits, to go through before an interview.
