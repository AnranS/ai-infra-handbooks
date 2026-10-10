# Chunked prefill

<p class="lead">Prefilling a 30,000-token prompt in one go can blow out memory with its activations, especially the MLP's intermediate results and the hidden state before the logits, and it also holds the GPU for a long time while everything else waits. Chunked prefill, from Sarathi-Serve, cuts a long prompt into chunks and computes one per step. mini-sglang's implementation is very compact: a <code>ChunkedReq</code> subclass for "a request that cannot decode", plus a few lines of budget logic in <code>PrefillAdder</code>.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Is chunked prefill mathematically equivalent to prefilling all at once? Why?
    2. A request is cut into 4 chunks. When chunk 2 computes attention, where does the KV of the first 16 tokens come from?
    3. Should the logits a chunk produces mid-way be sampled?
    4. In mini-sglang, what happens to other decoding requests while one request is being prefilled in chunks?

??? success "Answers (try it yourself first, then expand)"
    1. Equivalent: in causal attention each token depends only on the tokens before it, and a later chunk reads the KV earlier chunks already wrote through the page table, so the result is the same as computing it all at once (numerically there is only the tiny difference of floating-point ordering).
    2. From the KV pool: when the earlier chunks finished, their KV was written into the pool and the page table recorded where, so chunk 2's attention backend reads it back by the page table. This is the same situation as a prefix-cache hit.
    3. No: only when the last chunk finishes do the logits at the last position sample the first output token; a middle chunk's logits are thrown away.
    4. They pause: mini-sglang does only prefill or only decode per step, so decode waits during a chunked prefill. Production SGLang solves this by putting chunked prefill and decode in the same batch.

**Files you will write**: fill in `ChunkedReq`, `PrefillAdder._add_one_req` and `PrefillAdder.try_add_one` in `scheduler/prefill.py`.

@@tree@@

**This step's main**: `examples/ch10_chunked.py` — it uses only the files above; `python tools/steps.py check` rebuilds this tree chapter by chapter and runs it.

## Why it is equivalent {#为什么等价}

Attention is causal: position j depends only on tokens at positions ≤ j. Cut the prompt into `[0,16)`, `[16,32)` and so on, and once the first chunk is done its KV is already in the KV pool; the second chunk's queries read the first 32 positions' KV through the page table (16 from the previous chunk and 16 just written by this one), which is exactly what positions 16 to 31 would have seen in a single prefill. The MLP and RMSNorm are per-position anyway. So chunked prefill is mathematically equivalent to prefilling at once, only in a different order.

This is the same situation as the "prefix hit" of chapter 5: to the attention backend there is no difference at all between "the first 16 tokens are in the cache and we compute 16 this step" and "the first 16 tokens hit the prefix cache". So chunked prefill needs no change to the attention backend.

## ChunkedReq {#chunkedreq}

@@code python/minisgl/scheduler/prefill.py:ChunkedReq@@

A request in the middle of chunking is a subclass of `Req` that changes two things: it cannot be sampled (the logits the LM head computes this step belong to a middle position of the prompt and mean nothing) and it cannot join the decode set.

## The budget {#预算}

@@code python/minisgl/scheduler/prefill.py:PrefillAdder._add_one_req@@

Each prefill batch has a token budget, `token_budget` (that is `max_extend_tokens`, `--max-prefill-length` on the command line, 8192 by default). When a request is admitted, this chunk's size is the smaller of "budget left" and "request left"; if the chunk does not cover the rest of the prompt, a `ChunkedReq` is created, otherwise an ordinary `Req`.

`reserved_size` reserves for the request's **entire remaining** prompt plus `max_tokens`, not just this chunk, because the moment a request is first admitted all the space it will ever need is committed and no later chunk can get stuck for want of room.

@@code python/minisgl/scheduler/prefill.py:PrefillAdder.try_add_one@@

`PendingReq.chunked_req` remembers how far the chunking has got. When the request comes round again on the next step, it reuses the request slot and cache handle it already has and carries on from the previous `cached_len`, with no prefix matching or admission check.

Back in `PrefillManager.schedule_next_batch` from the last chapter, an unfinished chunked request goes back to the **head** of the waiting queue and continues first next step:

```python
self.pending_list = chunked_list + self.pending_list[len(reqs):]
```

The scheduler skips `ChunkedReq` when handling results (the `isinstance` check at the start of `_process_last_data` in chapter 7) and does not hand it to the prefix cache; its column is −1 when writing back to the token pool, since there is no sample to use as the next input.

## An example {#看一个例子}

@@diagram chunked-prefill the prefill steps of three requests with max_extend_tokens=16@@

@@code examples/ch10_chunked.py@@

@@output ch10_chunked@@

The 61-token prompt is cut into 16 + 16 + 16 + 13; the fourth step has 3 tokens of budget left, which go to uid1 behind it, making that a chunked request too; the fifth step finishes uid1's remaining 2 tokens and does uid2 in one go. Decode only starts once all three have prefilled. The final output matches Hugging Face prefilling the whole prompt at once.

## The cost {#代价}

Look at the trace above: there is no decode at all while the long prompt is being prefilled in chunks. That follows directly from "prefill first": as long as the waiting queue is non-empty the scheduler prefills, and the chunked request is always at the head. If other requests were decoding at the time, their next token would wait for every chunk to finish, and TPOT spikes.

What production SGLang and vLLM do is **put the prefill chunk and the decoding requests in one batch**: the decoding requests compute their one token as usual each step and the remaining budget goes to the prefill chunk. That way a long prompt never blocks decode, at the cost of two shapes in one batch, so CUDA Graph only covers pure decode steps and the scheduling logic gets more complex. mini-sglang chose "one kind per step" for simplicity.

!!! upstream "The official implementation"
    - @@upstream scheduler/prefill.py:ChunkedReq@@
    - @@upstream scheduler/prefill.py:PrefillAdder._add_one_req@@
    - The official docs/features.md warns that a chunk that is too small, say 128, noticeably hurts performance: every chunk runs the whole model and rereads the weights, and the smaller the chunk the less of the GPU is used.

## Tests {#测试}

@@code tests/test_ch10_chunked.py:test_chunked_prefill_is_exact_and_bounded@@

Every prefill batch uses exactly its 6-token budget (except the last), every prompt token is computed exactly once, and the output matches HF. Chapter 11's tests verify it again in the combination of chunking, overlap scheduling and a page size of 4.

!!! interview "How to explain it"
    On chunked prefill: causal attention makes chunking mathematically equivalent to prefilling at once, because a later chunk reads the KV earlier chunks already wrote through the page table, which is the same situation as a prefix-cache hit; middle chunks do not sample (their logits are discarded) and only the last chunk samples and joins decode. Each prefill batch is bounded by `max_extend_tokens`, and an unfinished chunked request returns to the head of the queue to continue first. The point is to bound the length and the memory peak of a single step; but mini-sglang does one kind per step, so decode still pauses during chunking, while production SGLang and vLLM mix chunked prefill with decode in one batch, letting decode advance every step, just more slowly.

## Exercises {#练习}

1. Implement a "mixed batch": in `_schedule_next_batch`, when the decode set is non-empty, put all the decoding requests into the batch first (one token each) and give the rest of the budget to prefill. What should `Batch.phase` be? What do the attention backend and the LM head need?
2. When a chunked request is aborted (the client disconnects), `PrefillManager.abort_req` returns its `chunked_req` and the scheduler then calls `_free_req_resources`. Why does aborting a non-chunked request that is still waiting need no release?
3. What happens if `max_extend_tokens` is 1?

??? success "Answers"
    1. You can keep `phase="prefill"` (there are requests with more than one query) and treat the decoding requests as "prefills" with `extend_len = 1`: the attention backend's metadata already supports any `cu_seqlens_q`, and the LM head takes each request's last position through `get_last_indices`, which for a decoding request is its only position. What mainly changes is the scheduler's bookkeeping (decoding requests take the normal `_process_last_data` path) and using CUDA Graph only on pure decode steps.
    2. A request that has not been admitted holds nothing, since the request slot, the KV pages and the cache lock are all acquired inside `_try_allocate_one`, so removing it from the queue is enough; a chunked request already holds all of them and must release them.
    3. Each prefill step computes 1 token, so a long prompt takes as many steps as it has tokens and rereads all the weights every step, which is extremely slow. The result is still correct.

## Summary {#小结}

- [x] Causal attention makes chunked prefill equivalent to prefilling at once: a later chunk reads earlier chunks' KV through the page table, the same situation as a prefix hit.
- [x] `ChunkedReq` neither samples nor joins decode; `PendingReq.chunked_req` remembers the progress and the next step continues with the same resources.
- [x] Each prefill batch is bounded by `max_extend_tokens`, and a chunked request returns to the head of the queue to continue first.
- [x] mini-sglang does one kind per step, so decode pauses while a long prompt is chunked; production builds solve this with a mixed batch.
