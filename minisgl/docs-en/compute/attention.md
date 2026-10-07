# The attention backend and a reference implementation

<p class="lead">Attention is the only computation in the engine that has to know the batch's structure: every request has a different length, the KV is scattered across pages of the pool, and some requests already have a cached prefix. mini-sglang puts all of that into the "attention backend": one <code>prepare_metadata</code> per batch turns the request information into whatever the kernel wants, and one <code>forward</code> per layer writes KV and computes attention. Upstream has three GPU backends, FlashInfer, FlashAttention and TensorRT-LLM; this chapter writes a PyTorch reference backend first to get the semantics straight, and chapter 17 wires up the GPU ones.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What does `cu_seqlens_q = [0, 6, 9, 10]` say?
    2. A request has 7 tokens of KV in total and 3 queries this step. How does the causal mask line up? Which keys can the first query see?
    3. Why is the metadata prepared once per batch rather than once per layer?
    4. Must the KV be written before or after attention is computed? Why?

??? success "Answers (try it yourself first, then expand)"
    1. This batch has 3 requests with 6, 3 and 1 queries each (as a prefix sum: request i's queries are `[cu[i], cu[i+1])`).
    2. Aligned to the bottom right: the 3 queries are the request's last 3 positions (4, 5, 6). The first query sits at position 4 and sees keys 0 through 4, five in all.
    3. Within one batch the variable-length information, the KV lengths and the page table are identical for every layer, so preparing once serves them all; preparing per layer is pure waste.
    4. Before: this step's new tokens need their own K and V in the attention too, since causal attention has every token see itself. So the KV goes into the pool first, and then all of it is read back for the computation.

**Files you will write**: `attention/base.py`, `attention/utils.py`, `attention/torch_backend.py`, `attention/__init__.py`.

## The interface {#接口}

@@code python/minisgl/attention/base.py:BaseAttnBackend@@

- `prepare_metadata(batch)`: the scheduler calls it once before each batch, and the result goes into `batch.attn_metadata`. The model has 28 layers whose attention shapes are identical, so the metadata only has to be prepared once; a GPU backend also copies data from the CPU to the GPU here, or runs FlashInfer's plan.
- `forward(q, k, v, layer_id, batch)`: called per layer. **Write first, read second**: put this step's new k and v into the KV pool, then read every request's full KV back out and compute attention. The order cannot be reversed, because this step's queries have to see this step's keys (the diagonal of the causal mask).
- The last three methods serve CUDA Graph and wait until chapter 18.

The metadata also provides `get_last_indices(bs)`: during prefill, each request's last token's index into the flattened q, for the LM head (chapter 2).

## Variable-length, paged attention with prefixes {#变长分页带前缀的注意力}

The tensors in a batch are one-dimensional arrays with the requests packed end to end. An example makes clear what attention needs to know: request A prefills 6 tokens from scratch; request B hits the prefix cache on its first 4 tokens and computes 3 this step; request C already has 8 tokens and decodes 1.

- **Where this step's queries are for each request**: `cu_seqlens_q = [0, 6, 9, 10]` (cumulative sequence lengths), so request i is `q[cu[i]:cu[i+1]]`;
- **How long each request's KV is**: `cache_seqlens = [6, 7, 9]`, each request's `device_len`;
- **Where each request's KV sits in the pool**: the matching row of the page table, taking its first `cache_seqlens[i]` locations;
@@diagram varlen-batch how a mixed batch splits its queries, where the KV comes from, and the causal mask@@

- **How the causal mask lines up**: this step's n queries are the **last** n positions of the sequence. With 7 tokens of KV and 3 queries, the first query sits at absolute position 4 and sees keys 0 through 4. This is "bottom-right alignment": draw the `[n, kv_len]` mask matrix and its diagonal lands in the bottom-right corner.

The reference backend is that description turned straight into code:

@@code python/minisgl/attention/torch_backend.py:TorchAttnBackend.forward@@

For each request: gather its full K and V from the pool through the page table, build the bottom-right-aligned causal mask, and call `scaled_dot_product_attention` (with `enable_gqa=True` so 16 q heads share 8 KV heads). It is a Python loop and it is not fast, but it is completely unambiguous and it runs on a CPU.

Preparing the metadata:

@@code python/minisgl/attention/torch_backend.py:TorchAttnBackend.prepare_metadata@@

The page table takes the first `max_seqlen_k` columns of each request's row, stacked into `[bs, max_seqlen_k]`, which is what the official FlashAttention backend does. It uses `batch.padded_reqs` rather than `batch.reqs`, because the dummy requests CUDA Graph pads in need metadata too.

Running the example above and comparing against "complete causal attention with no paging and no packing":

@@code examples/ch05_attention.py@@

@@output ch05_attention@@

A few observations:

- Request B's KV is not contiguous in the pool: the 4 tokens that hit the prefix cache are at 40-43 and the 3 new ones at 20-22. `out_loc` only covers this step's new tokens.
- The page table is cut at `max_seqlen_k = 9`, so shorter requests have irrelevant zeros after their data, which `cache_seqlens` keeps out of reach.
- `get_last_indices` gives `[5, 8, 9]`, the indices of the three requests' last queries in the flattened q.
- Against per-request complete causal attention the difference is at the level of float32 rounding.

## Creating a backend: one or a pair {#创建后端单个或组合}

@@code python/minisgl/attention/__init__.py:create_attention_backend@@

Backends are created by name through the registry. `"fa,fi"` means FlashAttention for prefill and FlashInfer for decode, dispatched by `HybridBackend` on `batch.is_prefill`:

@@code python/minisgl/attention/base.py:HybridBackend@@

CUDA Graph only captures decode, so the three graph-related methods only forward to the decode backend. With `--attn auto` the engine picks `torch` on a CPU, `fa,fi` on Hopper (SM90) and `fi` on other GPUs.

!!! diff "Difference from upstream: the reference backend"
    Upstream has no PyTorch backend (production SGLang has a `torch_native` backend that plays a similar role). Our `torch` backend uses the same metadata format as the official FlashAttention backend and implements the three CUDA Graph hooks the way that backend does, which is what lets chapter 18 verify CUDA Graph's data flow on a CPU. Upstream defaults to the `trtllm` backend on Blackwell (SM100), which we do not implement.

!!! upstream "The official implementation"
    - the interface and the combined backend: @@upstream attention/base.py:BaseAttnBackend@@, @@upstream attention/base.py:HybridBackend@@
    - the metadata format we follow: @@upstream attention/fa.py:FlashAttentionBackend.prepare_metadata@@
    - picking a backend automatically: @@upstream engine/engine.py:_adjust_config@@

## Tests {#测试}

@@code tests/test_ch05_attention.py:test_torch_backend_mixed_batch_with_prefix_hits@@

!!! interview "Answering in an interview"
    On the attention backend: keep the two jobs apart. One `prepare_metadata` per batch describes the variable lengths and the paging (`cu_seqlens_q` splits the queries per request, `cache_seqlens` gives the KV lengths, the page table gives the KV locations), and one `forward` per layer writes this step's K and V into the cache and then computes attention, because otherwise this step's tokens cannot see themselves. In a prefill with a prefix the causal mask aligns bottom right: with 7 tokens of KV and 3 queries, the first query sees the first 5 keys. The metadata is prepared once per batch because every layer sees the same batch structure. The reference backend is a Python loop plus SDPA and serves as the baseline for checking the FlashInfer and FlashAttention backends; prefill and decode may use different backends.

## Exercises {#练习}

1. The reference backend builds an `[n, kv_len]` mask per request. During decode n = 1 and the mask is all true. Write a fast path for decode: pad every request's KV to the maximum length as `[bs, max_len, H, D]` and do it in one batched SDPA.
2. What happens to a prefill if you forget to call `store_kv` first in `forward`? What about a decode?
3. What does `cu_seqlens_q` equal for "a prefill with no prefix hit"? And for "pure decode"? (Hint: look at the two fast paths in the FlashInfer backend in chapter 17.)

??? success "Answers"
    1. Build an `[bs, 1, max_len]` length mask from `cache_seqlens` (`arange(max_len) < len`) and gather the KV in one go with `page_table[:, :max_len]`; make sure the padding is masked out.
    2. During prefill this step's keys read stale data from the pool (uninitialized, or another request's), so the result is wrong; during decode the KV at the last position, the current token, is stale, which is wrong in the same way. Only "write first, read second" lets this step's token see itself.
    3. With no prefix hit each request's query count equals its KV length, so `cu_seqlens_q == cu_seqlens_k`; in pure decode each request has 1 query, so `cu_seqlens_q = [0, 1, 2, ..., bs]`.

## Summary {#小结}

- [x] The attention backend does two things: one `prepare_metadata` per batch, and one `forward` per layer that writes KV before computing attention.
- [x] The metadata describes variable lengths and paging: `cu_seqlens_q` splits the queries, `cache_seqlens` gives the KV lengths, the page table gives the KV locations, and the causal mask aligns bottom right.
- [x] The reference backend is a Python loop plus SDPA: clear semantics, runs on a CPU, and serves as the baseline for the GPU backends.
- [x] Backends are created by name, and prefill and decode may use different ones.
