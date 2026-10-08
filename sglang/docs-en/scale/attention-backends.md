# The attention backend matrix and paged KV

<p class="lead">The attention backend interface of September 2024 had two implementations; v0.5.0rc0 of August 2025 has over a dozen and the baseline commit more than thirty: FlashInfer, FlashAttention 3, FlashMLA, cutlass MLA, TRT-LLM MLA, AITER, Ascend, Intel, and every kind of hybrid and sparse attention. Over the same period the KV pool went from "one token per page" to a configurable page size (#4356 of March 2025), a change that pulled on the allocator, the radix tree, speculative decoding and PD disaggregation. This chapter covers how the backend matrix grew, why the page size waited until 2025, and how support for several kinds of hardware went from "AMD runs" to a <code>hardware_backend/</code> directory.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Going from 2 backends to over 30, what kinds do the new ones fall into? What does each solve?
    2. What does a page size of 1 cost? Which modules have to change to make it larger?
    3. Which kinds of method does today's `AttentionBackend` interface have that 2024's did not? Which feature does each serve?
    4. Where does the code for several kinds of hardware live? How does it relate to the attention backends?

??? success "Answers for the self-test (answer first, then open this)"
    1. By hardware (AITER for AMD, Ascend for Huawei's NPUs, Intel's CPU and XPU), by attention structure (MLA's FlashMLA / cutlass MLA / TRT-LLM MLA / FlashInfer MLA; the hybrid backends for hybrid models; sparse attention's NSA / DSA, double sparsity, dual chunk) and by kernel library (FlashInfer, FlashAttention 3, Triton, pure torch).
    2. Large metadata (one entry per token in the request table), an attention kernel addressing indirectly per token, and many external kernels (FA3, FlashMLA) designed around pages in the first place. The changes: a new `paged_allocator.py` allocating by page, the radix tree's keys page-aligned (matching only to a page boundary), `ScheduleBatch`'s allocation and freeing, copying candidate pages in speculative decoding, HiCache's backups by page, and PD's transfers by page.
    3. Finer graph methods (`init_forward_metadata_in_graph` / `out_graph`, an interruptible capture, an elastic recapture), speculative decoding's (`draft_extend_metadata_captured_in_graph`, `verify_mask`, `update_verify_buffers_to_fill_after_draft`), mixed batches (`forward_mixed`), shared-prefix reads (`shared_read_ends`, the chunked prefix) and sparse attention's index metadata (`get_indexer_metadata`).
    4. The `hardware_backend/` directory (95 files at the baseline commit) holds each vendor's platform adaptation (device queries, communication, kernel selection), and the attention backends are one layer of it: each kind of hardware usually corresponds to one or a few attention backends, with the platform layer deciding the default.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/attention-backends.webp is in Chinese; put it back once the English version exists -->

## The backend matrix {#后端矩阵}

```bash title="backends-per-tag.sh"
REF=${REF:-29f6d408c0}
for t in v0.4.0 v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s %2d 个：' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/layers/attention | grep -c '_backend\.py$')"
  git ls-tree -r --name-only "$t" -- python/sglang/srt/layers/attention | grep '_backend\.py$' | sed 's|.*/||; s|_backend\.py||' | tr '\n' ' ' | cut -c1-150; echo
done
```

```text title="output"
v0.4.0       4 个：double_sparsity flashinfer torch_native triton 

v0.4.6       8 个：base_attn double_sparsity flashattention flashinfer flashinfer_mla flashmla torch_native triton 

v0.5.0rc0   17 个：aiter ascend base_attn cutlass_mla double_sparsity dual_chunk_flashattention flashattention flashinfer flashinfer_mla flashmla hybrid_attn intel_amx t

29f6d408c0  34 个：aiter base_attn cutedsl_mla deepseek_v4 deepseek_v4_trtllm dots_hybrid dsa_topk paged_mqa_logits dsa flashattention flashinfer flashinfer_mla flashmla
```

A few milestones from the first half of 2025:

```bash title="backend-commits.sh"
for h in 36f6fc5093 c76040e31b a53fe428f9 b6944f97a6 5d7edc8e55 26c0f13126 20c90be23d 1c63e79756 e983e43248 51d25405a7; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2025-02-10  36f6fc5093  feat: enable ragged fa3 by default on hopper 12.4+ (#3442)
2025-03-12  c76040e31b  Support page size > 1 (#4356)
2025-03-17  a53fe428f9  Support FlashMLA backend (#4472)
2025-03-19  b6944f97a6  Support FlashMLA backend cuda graph (#4514)
2025-03-23  5d7edc8e55  Support FA3 as Attention backend by using `--attention-backend fa3` (#46
2025-03-27  26c0f13126  Support Page Size > 1 for FA3 (#4832)
2025-03-28  20c90be23d  [Feature] Support FA3 backend for MLA (#4831)
2025-03-31  1c63e79756  use fa3 in sgl-kernel (#4954)
2025-04-02  e983e43248  Add Eagle Speculative Decoding to FA3 Backend (#4951)
2025-03-04  51d25405a7  ROCm: update aiter and its usage to fused moe (bloat16, fp8, fp8 block-q
```

- **FlashAttention 3**: February first made FA3's ragged kernel the default for prefill on Hopper (#3442); #4680 of 23 March made it a complete backend, `--attention-backend fa3`; four days later came support for page sizes above 1 (#4832) and MLA (#4831); FA3 moved into sgl-kernel's build on 31 March (#4902, #4954); and EAGLE support arrived on 2 April (#4951). A backend goes from usable to fully featured in about two weeks.
- **MLA-specific backends**: FlashMLA (#4472, 17 March) gained CUDA graph support two days later; cutlass MLA, TRT-LLM MLA and FlashInfer MLA followed — MLA's decode kernel became everyone's competitive ground.
- **AMD**: AITER (#4053, #4075, early March) took over MoE and GEMM on ROCm; the attention backend `aiter_backend.py` already exists at v0.5.0rc0.

The interface itself grew too. The baseline commit's `base_attn_backend.py`:

```bash title="backend-interface.sh"
REF=${REF:-29f6d408c0}
echo "v0.4.0：$(git show v0.4.0:python/sglang/srt/layers/attention/__init__.py | grep -c '    def ') 个方法"
echo "今天：$(git show "$REF:python/sglang/srt/layers/attention/base_attn_backend.py" | grep -c '    def ') 个方法，其中和 cuda_graph 有关的 $(git show "$REF:python/sglang/srt/layers/attention/base_attn_backend.py" | grep '    def ' | grep -ci 'graph') 个、和投机解码有关的 $(git show "$REF:python/sglang/srt/layers/attention/base_attn_backend.py" | grep '    def ' | grep -ciE 'draft|verify') 个"
git show "$REF:python/sglang/srt/layers/attention/base_attn_backend.py" | grep '    def ' | sed 's/^ *def //; s/(.*//' | tr '\n' ' ' | cut -c1-400; echo
```

```text title="output"
v0.4.0：8 个方法
今天：22 个方法，其中和 cuda_graph 有关的 10 个、和投机解码有关的 3 个
max_of init_forward_metadata init_forward_metadata_out_graph init_forward_metadata_in_graph draft_extend_metadata_captured_in_graph shared_read_ends prepare_prefill_shared_read_snapshot prepare_full_cuda_graph_chunked_prefix init_cuda_graph_state validate_elastic_cuda_graph_recapture init_forward_metadata_for_breakable_cuda_graph_capture prepare_forward_metadata_for_breakable_cuda_graph_replay get
```

Against [chapter 10](../perf/restructure.md)'s eight methods, the additions correspond exactly to the later features: graph capture became segmentable and recapturable (piecewise CUDA graphs, elastic EP); speculative decoding's draft and verify want their own masks and buffers; mixed batches (`forward_mixed`); shared-prefix reads (several requests sharing one stretch of KV read it once); and sparse attention's index metadata (DeepSeek's DSA). The backend interface is a mirror of the whole system's features.

![Figure: the attention backends as a matrix of origin and hardware](../assets/figures/sgl-backend-matrix.svg){.aig-svg}

## The page size: a decision deferred a year {#页大小一个拖了一年的决定}

The first version's "one token per page" let the radix tree be cut anywhere ([chapter two](../origins/first-commit.md)). Its price was clear by 2024: one entry per token in the request table, kernels addressing indirectly per token, and external kernels mostly designed around pages. The H1 2025 roadmap listed "page size > 1 #4356" as an item, and it merged on 12 March:

```bash title="page-size-stat.sh"
git show --stat=100 --format='%ad  %an  %s' --date=short c76040e31b | grep -v '^$' | cut -c1-96
```

```text title="output"
2025-03-12  Lianmin Zheng  Support page size > 1 (#4356)
 python/sglang/srt/layers/quantization/fp8_kernel.py            |   2 +-
 python/sglang/srt/managers/schedule_batch.py                   | 271 ++++++++++++++++++--------
 python/sglang/srt/managers/schedule_policy.py                  |  15 +-
 python/sglang/srt/managers/scheduler.py                        | 125 +++++++------
 python/sglang/srt/managers/scheduler_output_processor_mixin.py |  13 +-
 python/sglang/srt/managers/tp_worker_overlap_thread.py         |   5 +-
 python/sglang/srt/mem_cache/base_prefix_cache.py               |  14 +-
 python/sglang/srt/mem_cache/chunk_cache.py                     |  56 ++----
 python/sglang/srt/mem_cache/hiradix_cache.py                   |   6 +-
 python/sglang/srt/mem_cache/memory_pool.py                     |  78 ++++++--
 python/sglang/srt/mem_cache/paged_allocator.py                 | 283 ++++++++++++++++++++++++++
 python/sglang/srt/mem_cache/radix_cache.py                     | 153 ++++++++++++----
 python/sglang/srt/model_executor/cuda_graph_runner.py          |   4 +
 python/sglang/srt/model_executor/forward_batch_info.py         |  20 +-
 python/sglang/srt/model_executor/model_runner.py               |  27 ++-
 python/sglang/srt/server_args.py                               |   2 +
 python/sglang/srt/utils.py                                     |   7 +
 test/srt/run_suite.py                                          |   1 +
 test/srt/test_dp_attention.py                                  |   6 +-
 test/srt/test_gptqmodel_dynamic.py                             |   1 +
 test/srt/test_mla_deepseek_v3.py                               |   2 +-
 test/srt/test_page_size.py                                     |  46 +++++
 test/srt/test_retract_decode.py                                |  18 ++
 23 files changed, 874 insertions(+), 281 deletions(-)
```

The three largest changes: a new `paged_allocator.py` (283 lines, allocating by page, with `alloc_extend` and `alloc_decode` handling "the last page still has room"), page-aligned keys in `radix_cache.py` (matching only to a page boundary, and nodes splitting by page too), and `schedule_batch.py`'s allocation and freeing. A run of "X plus page size > 1" followed: PD (#5561), FA3 (#4832), EAGLE (#4908), retraction (#4914), HiCache (#4581), and OOM with large pages (#4913) — every existing feature had to think about page boundaries again. Why wait until 2025? Because with a page size of 1 none of those features had to consider alignment at all, and building the features first and changing the pages once together was a better deal than making every feature more complicated from the start; the price is that the bill comes due all at once.

## Several kinds of hardware: from "AMD runs" to hardware_backend/ {#多硬件从amd-能跑到-hardware_backend}

```bash title="hardware-first.sh"
REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | grep -iE '^\S+\s+\S+\s+.*(\bamd\b|rocm)' | head -1 | cut -c1-96
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | grep -iE 'ascend|\bnpu\b' | head -1 | cut -c1-96
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | grep -iE '\bcpu backend\b|intel.*cpu|cpu.*intel' | head -1 | cut -c1-96
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | grep -iE '\bxpu\b' | head -1 | cut -c1-96
echo "hardware_backend/ 子目录：$(git ls-tree --name-only "$REF" python/sglang/srt/hardware_backend/ | sed 's|.*/||' | tr '\n' ' ')"
```

```text title="output"
2024-09-17  3a6e04185b  [Feature, Hardware] Enable SGLang on AMD GPUs via PyTorch for ROCm (#142
2025-05-07  00c2c1f08b  [Feature] Support for Ascend NPU backend (#3853)
2025-05-31  888cb175a6  Add intel_amx backend for Radix Attention for CPU (#6408)
2024-10-13  5d638c92f5  [Feature, Hardware] Enable SGLang on XPU GPUs via PyTorch (#1480)
hardware_backend/ 子目录：cpu gpu mlx musa npu xpu 
```

After AMD's first commit in September 2024, support for other hardware was for a long time `if is_hip()` branches scattered about; in 2025 the vendors (AMD's AITER, Ascend, Intel's CPU and XPU, and later TPUs through a separate sglang-jax) each contributed kernels and backends, and the branches multiplied. The baseline commit's `hardware_backend/` and `platforms/` gather the platform-dependent logic (device queries, the default backend's selection, the communication library, memory queries) in one place, with the attention backend only one layer of the platform adaptation. This is the shape the platform abstraction of the inference-systems handbook's [hardware-support chapter](serving://ops/platforms/) takes in SGLang.

## Design trade-offs {#设计取舍}

- **The backend interface grows with the features rather than being designed complete.** Eight methods served for a year; each new feature adds the methods it needs and the interface becomes a mirror of the features. The price is that implementing a new backend means facing over twenty methods (most with a default implementation or optional).
- **The page size came later.** Use a page size of 1 to keep every feature simple, then change the pages once in 2025 and adapt the features one by one.
- **External kernels first.** FA3, FlashMLA and FlashInfer are all external libraries and SGLang writes only the backend adaptation and the metadata; its own Triton backend guarantees that any platform can run and is the testbed for new features ([chapter 14](../perf/sgl-kernel.md)).

## What happened afterwards {#后来怎么样了}

- H2 2025: NSA and DSA sparse attention backends (DeepSeek-V3.2's indexer), hybrid backends for hybrid models (Mamba, linear attention plus full attention) with their memory pools, and a TRT-LLM backend for Blackwell.
- The page size's default is still 1 (the most compatible), while backends like MLA and FA3 are faster above 1, and the documentation recommends a value per backend.
- `hardware_backend/` has 95 files at the baseline commit and `platforms/` 8; TPUs use the separate sglang-jax repository (per the documentation of 2025-12).

## Exercises {#练习}

**1. Backend registration.** Find the attention backends' registry at the baseline commit (where a backend is chosen by name), count the supported names, and find which backends are only available on particular hardware.

??? success "A way to approach it"
    `git grep -n 'attention_backend' 29f6d408c0 -- python/sglang/srt/server_args.py` for the choices list, and `git grep -n 'def _get_attention_backend\|ATTENTION_BACKENDS' 29f6d408c0 -- python/sglang/srt/model_executor` for the registration and selection logic.

**2. What page alignment costs.** At a page size of 16, how many tokens of a 100-token prefix can hit on the radix tree? What happens to the rest?

??? success "Answer"
    Matching goes only to a page boundary: at most 96 (6 whole pages), with the remaining 4 recomputed. The larger the page, the coarser the prefix cache's granularity and the fewer tokens hit; this is the compromise between kernel efficiency and cache granularity.

**3. A backend's two weeks.** Use `git log --date=short --format='%ad %h %s' -- python/sglang/srt/layers/attention/flashattention_backend.py | tail -30` to see the FA3 backend's first 30 commits, and classify them as features, fixes or performance.

??? success "A way to approach it"
    You will see feature commits for the page size, MLA, EAGLE, sliding windows and CUDA graphs, plus a batch of accuracy and synchronisation fixes — the typical path by which a new backend joins the system.

!!! interview "How to explain it"
    "How does an inference engine support several attention kernels and several kinds of hardware?" — Explain through SGLang's interface: the backend interface leaves "how the metadata is prepared (including a CUDA graph's capture and replay)" and "how the forward pass computes" to the implementation, so a new backend only adds a file; and the interface grows with the features (speculative decoding, mixed batches and sparse indexing each add methods). The page-size story illustrates the "simple first, unified later" trade-off. Several kinds of hardware rest on a platform layer gathering the device-dependent logic, with the attention backend one layer of it.

## Summary {#小结}

- [x] From 2 backends to over 30, in three kinds: by hardware, by attention structure and by kernel library; the interface's method count grew from 8 to 22, mirroring the system's features.
- [x] Page size > 1 (#4356, 2025-03-12): a new allocator, a page-aligned tree, and every feature adapted afterwards; waiting until 2025 was a deliberate trade-off.
- [x] Support for several kinds of hardware went from scattered branches to `hardware_backend/` and `platforms/`, with the attention backend one layer of the platform adaptation.
