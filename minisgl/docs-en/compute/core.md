# Core data structures: Req, Batch, Context

<p class="lead">Only a few things flow through the whole system: a request <code>Req</code>, a batch of them <code>Batch</code>, and the global <code>Context</code> that holds "which batch is being computed right now". They all live in one 136-line file, yet they decide how every later module is written. This chapter writes them, with the focus on the length fields on <code>Req</code>, because the scheduler's entire bookkeeping is built on them.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. A request has a prompt of 6 tokens, its first 2 tokens hit the prefix cache, and `max_tokens=3`. How many tokens does the prefill step compute? What is `device_len` after prefill?
    2. Why is `Req` declared `@dataclass(eq=False)`?
    3. What is the difference between a `Batch`'s `reqs` and its `padded_reqs`?
    4. How does each model layer get hold of the current batch's positions?

??? success "Answers (try it yourself first, then expand)"
    1. Four (`extend_len = device_len − cached_len = 6 − 2`). `device_len` is 6 for that step; after the forward pass `complete_one()` moves `cached_len` to 6 and `device_len` to 7, so the next decode step computes position 7.
    2. So that `Req` compares and hashes by identity. The scheduler puts requests into a `set`, and two different requests can have every field equal (two identical prompts arriving at once), which field-wise comparison would collapse into one.
    3. `reqs` are the real requests; `padded_reqs` also includes the dummy requests used to pad the batch size when CUDA Graph is on.
    4. From the global `Context`: the scheduler fills in tensors like `positions` while preparing the batch, and each layer reads what it needs.

**Files you will write**: `minisgl/core.py`.

## SamplingParams {#samplingparams}

@@code python/minisgl/core.py:SamplingParams@@

`is_greedy` lets the sampler take a fast path: when the whole batch is greedy it goes straight to `argmax` without a softmax. Note that `top_k == 1` counts as greedy but `top_p < 1` does not, because even at temperature 0 it still goes through the top-p filtering (the result is still deterministic, just a different code path).

## Req: the three lengths of a request {#req请求的三个长度}

A request's state inside the engine is fully described by three lengths:

@@diagram req-lengths the same request's three lengths at admission, after prefill, and after one decode step@@

- `cached_len`: how many leading tokens already **have their KV in the cache** (computed earlier, or hit in the prefix cache);
- `device_len`: how many tokens will have KV in the cache **after this forward pass**;
- `max_device_len = prompt length + max_tokens`: the longest it can grow.

Everything the scheduler needs follows from these:

- `extend_len = device_len - cached_len`: how many tokens go into the model this step. During prefill it is the part of the prompt that missed the cache; during decode it is 1;
- `remain_len = max_device_len - device_len`: how many more tokens it may generate;
- `can_decode = remain_len > 0`: whether another decode step is allowed.

After each forward pass `complete_one()` advances the state: this step's tokens are now cached (`cached_len = device_len`) and the next step computes one more position (`device_len += 1`). The sampled token is appended by the scheduler to the CPU-side `input_ids` with `append_host`.

@@code python/minisgl/core.py:Req@@

Following the request from the self-test through:

@@code examples/ch01_req.py@@

@@output ch01_req@@

A few details worth noticing:

- The prefill step has `extend_len = 4`: the 2 tokens that hit do not need computing again.
- The KV of the third generated token (the one sampled at decode 2) is never computed: `device_len = 9` while `cached_len = 8`. The last token only has to be emitted, never fed back in.
- `input_ids` lives on the CPU (there is an assertion in `__post_init__`). The GPU keeps its own copy in the `token_pool` of chapter 4; the CPU copy is used for prefix-cache insertion and for deciding when a request is finished.
- `@dataclass(eq=False)` makes `Req` compare and hash by identity. The scheduler puts requests into a `set`, and two different requests can have every field equal, for instance two identical prompts arriving at once.

## Batch: who fills which field {#batch谁来填哪个字段}

@@code python/minisgl/core.py:Batch@@

`Batch` only takes `reqs` and `phase` in its constructor; the other fields are filled by different modules at different points in scheduling:

| Field | Filled by | What it holds |
| --- | --- | --- |
| `padded_reqs` | `GraphRunner.pad_batch` (chapter 18) | to use CUDA Graph, a decode batch is padded with dummy requests up to a captured batch size; without graphs it equals `reqs` |
| `positions` | the scheduler's `_prepare_batch` | this step's position for each token, `[cached_len, device_len)`, concatenated into one dimension |
| `out_loc` | the scheduler's `_prepare_batch` | where in the KV pool each of this step's tokens writes its KV |
| `input_ids` | the scheduler's `_forward` | this step's input tokens, gathered by index from the `token_pool` on the GPU |
| `attn_metadata` | the attention backend's `prepare_metadata` | the variable-length and paging information the attention kernel needs |

Every tensor **packs the batch's requests end to end into one dimension** (`[total tokens]`), with no padding. This is standard in modern inference engines: prefill and decode requests differ wildly in length, and padding would waste a lot of compute. The boundaries between requests are described by `cu_seqlens` in the attention metadata (chapter 5).

## Context: the global "current batch" {#context全局的当前-batch}

@@code python/minisgl/core.py:Context@@

`Context` holds the things there is exactly one of per process: the page table, the attention backend, the MoE backend, the KV pool, and "the batch being computed right now". `forward_batch` is a context manager that sets the current batch only inside the `with` block, clears it on exit, and refuses to nest.

@@code python/minisgl/core.py:get_global_ctx@@

So the model code can be written as `model.forward()`, with no arguments. The attention layer reads `get_global_ctx().batch.positions` when it needs positions, and the LM head reads `batch.is_prefill` when it needs to know whether this is prefill. The upside is that no arguments are threaded through the model and every layer's signature stays clean; the price is an implicit dependency, since calling the model outside a `forward_batch` context fails an assertion outright.

!!! upstream "The official implementation"
    The whole of @@upstream core.py@@. Our version has exactly the same fields and methods plus one extra `reset_global_ctx()` for tests: the official `set_global_ctx` refuses to be called twice, and tests create engines repeatedly in one process, so they need to clear it first.

## Tests {#测试}

@@code tests/test_ch01_core.py:test_req_lengths_through_prefill_and_decode@@

The other two tests in `tests/test_ch01_core.py` check the edge cases of `is_greedy`, and that `forward_batch` cannot nest and that the current batch is unreachable after it exits.

!!! interview "How to explain it"
    On request state: mini-sglang's `Req` describes its state with just three lengths, `cached_len` (already in the cache), `device_len` (in the cache after this step) and `max_device_len` (the ceiling), and this step computes `extend_len = device_len - cached_len`, of which prefill, chunked prefill and decode are all special cases. With a prompt of 6 and a 2-token prefix hit, the prefill step computes 4 tokens. The tensors in a `Batch` are one-dimensional arrays with the requests packed end to end and no padding (CUDA Graph pads the batch size through `padded_reqs`), and every model layer takes the current batch's positions and attention metadata from the global `Context`. `Req` uses `eq=False` so that it compares and hashes by identity, which is what lets it go into a set or serve as a dictionary key.

## Exercises {#练习}

1. A request has a prompt of 100 tokens, `max_tokens=50`, and its first 64 tokens hit the cache. Write down `cached_len`, `device_len` and `remain_len` at admission, after prefill, and after the 10th decode step.
2. If `Req` used the default `eq=True`, what would go wrong with the scheduler's `running_reqs: Set[Req]`? (Hint: a dataclass with `eq=True` is unhashable.)
3. Why does `Req.__post_init__` require `cached_len < device_len` rather than `<=`? When could that be violated?

??? success "Answers"
    1. At admission `(64, 100, 50)`; after prefill `(100, 101, 49)`; after the 10th decode step `(110, 111, 39)`.
    2. With `eq=True` (the default) and no `frozen=True`, `@dataclass` sets `__hash__` to `None`, so `Req` cannot go into a `set`; and even if it could be hashed, two different requests with identical contents would be treated as one. `eq=False` keeps `object`'s identity comparison and hashing.
    3. Every step has to compute at least one token: even when the whole prompt is cached, the hidden state at the last position is still needed for the logits. So the scheduler deliberately leaves the last token unmatched when matching a prefix (the `input_len - 1` in `match_req`, chapter 8), which keeps `cached_len < device_len`.

## Summary {#小结}

- [x] `Req` describes its state with three lengths: `cached_len` (already cached), `device_len` (cached after this step) and `max_device_len` (the ceiling); `extend_len`, `remain_len` and `can_decode` all derive from them.
- [x] `complete_one()` advances the state after each forward pass, and the scheduler appends the new token to the CPU-side `input_ids` with `append_host`.
- [x] The tensors in a `Batch` are one-dimensional arrays with the requests packed end to end and no padding; the fields are filled by the scheduler, the attention backend and the GraphRunner at different points.
- [x] The model reads the current batch from the global `Context`, so `forward()` needs no arguments.
