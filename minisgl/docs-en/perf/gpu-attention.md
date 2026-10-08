# GPU attention backends: FlashInfer and FlashAttention

<p class="lead">The reference backend computes attention one request at a time in a Python loop, which is clear but slow. On a GPU mini-sglang uses two specialized libraries: FlashInfer (very strong at decode and good at prefill) and FlashAttention 3 (the fastest prefill on Hopper). Their interfaces differ, since FlashInfer plans before it runs and FlashAttention takes a page-number table directly, but both express the metadata of chapter 5. This chapter implements both backends the way upstream does and verifies the argument semantics on a CPU through PyTorch fakes with the same interfaces.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What do FlashInfer's `plan` and `run` do? Why does plan happen once per batch?
    2. mini-sglang treats the KV pool as page size 1 for FlashInfer. What goes into `indices`? Why is `last_page_len` always 1?
    3. Does FlashAttention's `page_table` hold token locations or page numbers? How is it derived from mini-sglang's global page table?
    4. How do you verify a backend that only runs on a GPU, on a CPU?

??? success "Answers (try it yourself first, then expand)"
    1. `plan` takes this batch's metadata (each request's length, the page table) and works out the schedule: it divides the work, allocates temporary buffers and copies the metadata to the GPU. `run` then executes that plan per layer. Every layer of a batch has the same metadata, so plan runs once, deferred until the first layer.
    2. `indices` holds the pool location of each token's KV (with a page size of 1 the page number is the token location), that is, one row of the page table; each page holds one token, so the last page is always full and `last_page_len` is always 1.
    3. Page numbers. Take every page_size-th location from the global page table (which stores token locations) and divide by the page size to get each page's number.
    4. Write a fake with the same interface as the real library: translate both libraries' argument semantics into one reference implementation (Python plus SDPA), run the backend code on a CPU, and require the result to match the reference backend exactly.

**Files you will write**: `attention/fi.py`, `attention/fa.py`, plus `tests/fakes/flashinfer/` and `tests/fakes/sgl_kernel/` for the tests.

## FlashInfer: plan and run {#flashinferplan-与-run}

FlashInfer provides a "batch prefill" and a "batch decode" wrapper for paged KV caches. Using them takes two steps:

- `plan(...)`: pass in this batch's shape information, namely each request's query range (`qo_indptr`), which pages make up each request's KV (`kv_indptr` dividing `kv_indices`), and how full the last page is (`last_page_len`). From that FlashInfer works out the division of labour on the CPU (which thread block handles which stretch of KV) and copies it to the GPU asynchronously;
- `run(q, paged_kv_cache)`: called once per layer, computing according to the plan.

All 28 layers have the same shapes, so plan is needed only once. mini-sglang defers it into the first layer's `forward` (`_initialize_metadata_once`) rather than doing it in `prepare_metadata`:

@@code python/minisgl/attention/fi.py:FlashInferBackend.forward@@

@@code python/minisgl/attention/fi.py:FlashInferBackend.prepare_metadata@@

A few designs worth noticing:

- **treating the KV pool as page size 1**: `forward` views the pool as `[tokens, 1, heads, dim]`, so one row of the global page table is `kv_indices` and `last_page_len` is always 1. Whatever the system's page size, the FlashInfer backend never has to convert anything, which is the payoff of chapter 4's "the page table stores token locations";
- **two fast paths**: in pure decode `qo_indptr = [0, 1, …, bs]`; in a prefill with no prefix hit each request's query count equals its KV length, so `qo_indptr` simply reuses `kv_indptr`;
- **pinned memory and events**: plan reuses a block of pinned memory for its asynchronous copy, so the previous copy has to finish before the next plan modifies it (`last_event.synchronize()`);
- **Tensor Cores**: when the GQA group is 4 or more, decode is effectively a series of small matmuls and the Tensor Core decode kernel is faster.

@@code python/minisgl/attention/fi.py:FlashInferBackend._initialize_metadata_once@@

## FlashAttention: a page-number table {#flashattention页号表}

`flash_attn_with_kvcache` in `sgl_kernel` (FlashAttention 3) reads the paged KV pool directly and wants `page_table` (each request's **page numbers**), `cache_seqlens` (each request's KV length) and `cu_seqlens_q` (the variable-length query split). There is no plan step.

@@code python/minisgl/attention/fa.py:FlashAttentionBackend.prepare_metadata@@

The global page table stores token locations, so every page_size-th location is a page's start and dividing by the page size gives its number: `page_table[r, :max_len:page_size] // page_size`.

## Verifying on a CPU {#在-cpu-上验证}

Both libraries only run on NVIDIA GPUs. To check that the backend code passes the right arguments, we wrote a fake with the same interface for each, under `tests/fakes/`, placed at the front of the import path during tests:

@@code tests/fakes/flashinfer/__init__.py:BatchPrefillWithPagedKVCacheWrapper@@

@@code tests/fakes/sgl_kernel/flash_attn.py:flash_attn_with_kvcache@@

Both translate their arguments into "each request's query range, KV page list and KV length" and hand them to one reference implementation:

@@code tests/fakes/_fake_ref.py:paged_attention@@

So any argument the backend gets wrong, such as an `indptr` missing its leading 0, a page number not divided by the page size, or a mistaken `qo_indptr` during decode, makes the fake's result differ from the reference backend and the test fail. These fakes are also the shortest path to understanding what FlashInfer's and FlashAttention's arguments mean.

@@code examples/ch17_gpu_attention.py@@

@@output ch17_gpu_attention@@

Two requests: request 0 hits its first 3 tokens and computes 2 this step; request 1 has 5 in the cache and computes 1. FlashInfer receives token-level `kv_indices` (16-20 and 40-45) with `last_page_len` all 1; FlashAttention, with a page size of 4, receives page numbers (location 16 is page 4, location 40 is page 10). Both backends' logits are identical to the reference backend's.

!!! upstream "The official implementation"
    - @@upstream attention/fi.py:FlashInferBackend@@
    - @@upstream attention/fa.py:FlashAttentionBackend@@
    - Upstream also has a TensorRT-LLM backend, `attention/trtllm.py` (the default on Blackwell, supporting only page sizes 16, 32 and 64), which we do not implement.

    Our two backends follow upstream's logic exactly and only replace `pin_memory=True` and `torch.cuda.Event` with the device abstraction's functions.

## Tests {#测试}

@@code tests/test_ch17_gpu_backends.py:test_backend_logits_match_torch_backend@@

`tests/test_ch17_gpu_backends.py` also checks that plan runs once per batch, and compares end to end against Hugging Face under overlap scheduling and chunked prefill in three configurations: FlashInfer (page size 1), FlashAttention (page size 4) and the `fa,fi` pair.

!!! interview "How to explain it"
    On GPU attention backends: FlashInfer splits into plan and run, where plan works out the task division and temporary buffers from this batch's length distribution (CPU work, once per batch, deferred to the first layer) and run is called per layer; mini-sglang treats the KV pool as page size 1, so a page-table row is `kv_indices` and `last_page_len` is always 1. FlashAttention takes a page-number table, the KV lengths and `cu_seqlens_q` directly, with page numbers obtained by taking every "page size"-th location from the global page table and dividing. Without a GPU, fakes with the same interfaces translate both libraries' argument semantics into a reference implementation and verify that the backends build their metadata correctly.

## Exercises {#练习}

1. To have FlashInfer use the real page size (16, say) instead of flattening to 1, what changes in `prepare_metadata`? How is `last_page_len` computed?
2. Deliberately break the fake FlashInfer's interpretation of `qo_indptr` (ignore the leading 0, say). Which test fails?
3. Why does CUDA Graph only ever deal with the decode backend, and why does FlashAttention's prefill not need it?

??? success "Answers"
    1. `kv_indices` becomes page numbers: `page_table[r, :device_len:ps] // ps`; `kv_indptr` accumulates page counts (`ceil(device_len / ps)`); `last_page_len = device_len - (pages - 1) * ps`; and `forward` stops viewing the pool as page size 1 and passes the `[pages, ps, heads, dim]` pool directly.
    2. `test_backend_logits_match_torch_backend[fi]` and every end-to-end test using `fi` fail, which is exactly what the fakes are for.
    3. A prefill batch's shape (token count, per-request lengths) varies endlessly and no graph can be captured for every shape; prefill also does a lot of compute, so kernel-launch overhead is a small share and graphs are unnecessary. A decode batch's shape depends only on the batch size and each step does very little compute, so launch overhead is a large share.

## Summary {#小结}

- [x] FlashInfer: plan (once per batch, deferred to the first layer) plus run (per layer); mini-sglang treats the KV pool as page size 1, so a page-table row is `kv_indices`.
- [x] FlashAttention: takes a page-number table, the KV lengths and `cu_seqlens_q` directly, with page numbers taken every page_size-th location from the global page table and divided.
- [x] Fakes with the same interfaces translate both libraries' argument semantics into one reference implementation, verifying the backend code on a CPU against results identical to the reference backend.
