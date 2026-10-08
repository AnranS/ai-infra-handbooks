# The KV pool, the page table and the token pool

<p class="lead">At startup the engine allocates three blocks of memory once and for all: the KV pool that holds every request's K and V, the page table that records where in the pool each token's KV lives, and the token pool that records which token sits at each position of each request. The attention backend, the scheduler and overlap scheduling all work around these three blocks. This chapter builds them and explains one design that is particular to mini-sglang: the page table always stores per-token locations, whatever the page size.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How many bytes does one token's KV take for Qwen3-0.6B in bf16? How many tokens fit in 40 GiB?
    2. What problem does paging (page size > 1) solve? Is a bigger page always better?
    3. Why is a copy of the input tokens kept on the GPU (the token pool) instead of sending them from the CPU each step?
    4. Why does the KV pool allocate one extra page?

??? success "Answers (try it yourself first, then expand)"
    1. 2 × 28 layers × 8 KV heads × 128 dims × 2 bytes = 114688 bytes (112 KiB); 40 GiB holds about 374,000 tokens (40 × 1024 × 1024 ÷ 112).
    2. Allocating and managing by the page makes the block table shorter and the number of allocations and frees smaller, and it also makes sharing a prefix or transferring data page by page easy. Bigger is not always better: the last page leaves more internal fragmentation, and the prefix cache can only cache whole pages, which coarsens the hit granularity.
    3. The sampled token is written straight back into the token pool on the GPU, and the next step's input is read from there, which keeps the CPU off the critical path. That is also what makes overlap scheduling possible.
    4. For the dummy request that pads the batch for CUDA Graph: it writes KV too, and the extra last page is where it writes, so it never steps on a real request's data.

**Files you will write**: `kvcache/base.py`, `kvcache/mha_pool.py`, `kvcache/naive_cache.py`, `kvcache/__init__.py`, `scheduler/table.py`.

## The KV pool: one preallocated block of memory {#kv-池一整块预分配的显存}

During inference each request's KV cache grows as it generates, and requests come and go. Letting each request allocate its own memory fragments it quickly. Every modern inference engine does the same thing: **allocate the remaining memory into one big pool at startup and manage it in fixed-size pages**, handing out pages on demand and taking them back when done. That is the idea behind PagedAttention (the concept is in the [Inference Systems handbook's paged KV cache chapter](serving://engine/paged-kv/)).

@@code python/minisgl/kvcache/mha_pool.py:MHAKVCache@@

The layout is `[2, layers, pages, page_size, this rank's KV heads, head_dim]`:

- dimension 0 separates K from V, and `k_cache(i)` and `v_cache(i)` give layer i's `[pages, page_size, heads, head_dim]`;
- flatten "pages × page_size" and row j is the location of the pool's j-th token, which is what `_storage_shape` describes;
- `store_kv` writes this step's new K and V by `out_loc`, each new token's location in the pool. The attention backend calls it before computing attention.

@@code python/minisgl/kernel/torch_ops.py:store_cache@@

This is pure indexed movement, done on the GPU by the custom kernel of chapter 19: one warp per token, 16 bytes per move.

@@code examples/ch04_kvcache.py@@

@@output ch04_kvcache@@

Qwen3-0.6B uses GQA (16 q heads, 8 KV heads), so one token's KV is 112 KiB. The weights are only about 1.2 GB in bf16, so on an 80 GB card almost all the memory left after weights and activations goes to the KV pool. How many requests can be served at once, and how long each can be, is decided by the size of that pool.

**Why the extra page?** The engine keeps a dummy request for CUDA Graph (chapter 18), and it writes KV too when it pads the batch. The extra last page is where it writes, so it never steps on a real request's data.

## The page table: locations per token {#page-table按-token-存位置}

@@diagram kv-layout the KV pool, the page table and the token pool: two requests' KV scattered across pages@@

The page table is a two-dimensional int32 tensor of shape `[max_running_req + 1, max_seq_len rounded up to 32]`, one row per running request (the last row for the dummy), where `page_table[r, j]` is the location in the pool of that request's j-th token's KV.

One thing is particular to mini-sglang: **whatever the page size, the page table stores token locations rather than page numbers**. In the example above the page size is 16 and request 0 occupies pages 2 and 5, so its page-table row is `[32, 33, …, 47, 80, 81, 82, 83]`, not `[2, 5]`.

The benefit is simplicity:

- when writing KV, `out_loc = page_table[row, cached_len:device_len]` gives each new token's location directly, with no "page number × page size + offset" arithmetic;
- the FlashInfer backend treats the KV pool as if page size were 1, so one row of the page table is exactly the `indices` it wants;
- FlashAttention wants page numbers, so it takes every page_size-th location and divides: `page_table[r, ::page_size] // page_size` (chapter 17).

The price is a page table page_size times larger, but it is only int32, at most tens of MB. Rounding to 32 int32 values (128 bytes) aligns the start of each row, which the GPU reads more efficiently.

A page size above 1 matters for **the granularity of prefix caching and allocation**: the radix cache only caches whole pages (chapter 9) and the allocator hands out a whole page at a time (chapter 8), so larger pages mean less management overhead, at the cost of half a page wasted on average at the end and prefix matching that can only align to pages. Some kernels, such as TensorRT-LLM's attention, only support particular page sizes.

## The token pool: inputs live on the device too {#token-pool输入也放在设备上}

@@code python/minisgl/scheduler/table.py:TableManager@@

`TableManager` manages the page table's rows: a request gets a row when it is admitted and returns it when it finishes. It also owns the **token pool**, an int32 tensor of the same shape as the page table, where `token_pool[r, j]` is the token id at that request's j-th position.

Why keep another copy of the tokens on the device? Because this step's input is **the previous step's sample**. Once the engine has `next_tokens`, the scheduler runs `token_pool[row, device_len] = next_tokens` right on the device, and when the next batch is assembled `input_ids = token_pool[row, positions]` also happens on the device. The CPU never needs to know what the new token is. This is exactly what overlap scheduling (chapter 11) relies on: while the GPU computes step N+1, the CPU may not yet have step N's results.

The dummy request also reads its input from the token pool, so the pool is initialized to a valid token id (0) rather than random values.

## The prefix-cache interface {#前缀缓存的接口}

The KV pool only stores; it has no idea which location belongs to whom. What decides which locations can be reused is the **prefix cache**, whose interface is in `kvcache/base.py`:

@@code python/minisgl/kvcache/base.py:BasePrefixCache@@

This chapter implements the simplest one, `NaivePrefixCache`: it never hits and keeps nothing on insert, so a request's pages are all freed when it ends.

@@code python/minisgl/kvcache/naive_cache.py:NaivePrefixCache@@

The real radix cache arrives in chapter 9. Both are created by name (`naive`, `radix`) through the `SUPPORTED_CACHE_MANAGER` registry.

!!! upstream "The official implementation"
    - the KV pool: @@upstream kvcache/mha_pool.py:MHAKVCache@@, with the KV writes done by a custom CUDA kernel (`kernel/csrc/jit/store.cu`)
    - creating and aligning the page table: the "Page table initialization" section in @@upstream engine/engine.py:Engine.__init__@@, whose comment reads "aligned to 128 bytes; store raw locations instead of pages"; the `Context` in `core.py` also notes "this table always treat page_size = 1"
    - the token pool: @@upstream scheduler/table.py:TableManager@@

## Tests {#测试}

@@code tests/test_ch04_kvcache.py:test_kv_pool_layout_and_store@@

The second test checks KV-head replication under tensor parallelism: 2 KV heads across 4 ranks, one head per rank (chapter 16).

!!! interview "How to explain it"
    On the KV pool: allocate one block of memory at startup, `[2, layers, pages, page_size, KV heads, head_dim]`, and hand it out in pages, which avoids runtime allocation and fragmentation; one token's KV is 2 × layers × KV heads × head_dim × 2 bytes, or 112 KB for Qwen3-0.6B. The page table has one row per request and stores KV locations per token, and an attention backend that wants page numbers converts them itself. Paging lets memory grow on demand and makes prefix sharing possible; bigger pages mean less metadata and more efficient kernels but more waste at the tail and coarser prefix reuse. The token pool keeps the input tokens on the device too, so the sample is written back and read as the next input without the CPU on the critical path. One extra page is allocated for the dummy request used for padding.

## Exercises {#练习}

1. How large is one token's KV for Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128) in bf16? After 16 GB of weights on an 80 GB card, how many tokens fit?
2. If the page table stored page numbers instead, how would `out_loc` be computed when writing KV? Write the PyTorch expression.
3. With a page size of 16, how many pages does a 17-token request take? How much space does an average request waste?

??? success "Answers"
    1. 2 × 32 × 8 × 128 × 2 = 131072 bytes (128 KiB); with about 64 GiB usable that is roughly 520,000 tokens (in practice activations and CUDA Graph memory come off the top too).
    2. `page_ids[pos // page_size] * page_size + pos % page_size`, where `pos` is `[cached_len, device_len)`.
    3. 2 pages; the last page is half used on average, so the waste averages about page_size / 2 = 8 tokens.

## Summary {#小结}

- [x] The KV pool is one big tensor `[2, layers, pages, page_size, heads, dim]` allocated at startup and handed out in pages, with one extra page for the dummy request.
- [x] The page table has one row per request and stores KV locations per token regardless of page size; a backend that wants page numbers converts them itself.
- [x] The token pool keeps each request's token at each position on the device, so the sample is written back directly and the next step's input read directly, keeping the CPU off the critical path.
- [x] The prefix cache decides which locations can be reused; for now it is the naive version that never hits.
