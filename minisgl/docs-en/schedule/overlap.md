# Overlap scheduling: hiding the CPU cost

<p class="lead">One decode step of a small model takes the GPU a few milliseconds; what the scheduler has to do on the CPU each step, namely receive messages, assemble the batch, prepare metadata, handle the previous step's results and send messages, takes a few milliseconds too. Taking turns leaves the GPU waiting on the CPU almost half the time. Overlap scheduling, from NanoFlow and introduced in SGLang 0.4, lets the CPU handle step N's results while the GPU computes step N+1. mini-sglang's implementation adds barely a dozen lines, but it changes <em>when you know what</em>, which is where the four problems this reimplementation found come from.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. When step N+1 is launched the CPU does not yet know what step N sampled. Where does step N+1's input come from?
    2. A request samples EOS at step N. Can it appear in step N+1?
    3. Why do the scheduler and the engine need two different CUDA streams?
    4. Under overlap scheduling, when step N's results are handled, which step does `req.device_len` reflect?

??? success "Answers (try it yourself first, then expand)"
    1. From the token pool on the GPU: step N's sample is written straight into the token pool on the GPU, and step N+1's forward pass reads its input from there, so the CPU never has to know what it is.
    2. It can: when step N+1 is launched the CPU has not handled step N's results and does not know it has finished, so it is scheduled one more time; the extra token is discarded when the results are handled.
    3. So that the scheduler's work preparing the next step (copying metadata, assembling the batch) runs in parallel with the engine's compute, with events establishing the dependencies; on one stream they would be serialized again.
    4. After step N+1: the request state on the CPU was advanced at launch time, so it runs one step ahead of the results being handled.

**Files you will write**: `overlap_loop` and `run_forever` in `scheduler/scheduler.py`, plus the overlap-related parts of `_process_last_data`, `_free_req_resources` and `_process_one_msg`.

@@tree@@

**This step's main**: `examples/ch11_overlap.py` — it uses only the files above; `python tools/steps.py check` rebuilds this tree chapter by chapter and runs it.

@@video overlap Animation: overlap scheduling and the four problems it brings (about 2 minutes, Chinese narration and subtitles)@@

## The two loops {#两种循环}

@@code python/minisgl/scheduler/scheduler.py:Scheduler.overlap_loop@@

Against `normal_loop` there is one difference: **launch this step first, then handle the previous one**. `overlap_loop` takes the previous step's data and returns this step's, and `run_forever` strings them together:

@@code python/minisgl/scheduler/scheduler.py:Scheduler.run_forever@@

With one request and 3 tokens, the order of events (the first half of the example):

<!-- i18n:diagram 703d69cf26 -->
```text
normal loop:   launch 1  handle 1  launch 2  handle 2  launch 3  handle 3
overlap loop:  launch 1  launch 2  handle 1  launch 3  handle 2  handle 3
```

On a GPU, "launch" only queues kernels on a stream and returns immediately; "handle" has to wait for the result to be copied back to the CPU. In the overlap loop, by the time step 1 is handled the GPU is already computing step 2, and the CPU's cost hides behind the GPU's compute:

@@diagram overlap-timeline the CPU / GPU timeline of the normal loop and the overlap loop@@

## Three prerequisites {#三个前提}

All of this rests on three designs laid down in earlier chapters.

**First, the next step's input never passes through the CPU.** Step N+1's decode input is the token step N sampled, which the CPU does not know when step N+1 is launched. But `_forward` wrote the sample straight into the token pool's next position on the GPU, and step N+1 reads its input from there (the `write_tuple` and `input_tuple` of chapter 7). A GPU executes a stream in order, so the read always comes after the write.

**Second, the state scheduling needs is advanced early on the CPU.** `Engine.forward_batch` calls `complete_one()` on each request right after launching, so by the time step N+1 is scheduled the request's `cached_len` and `device_len` are already the "after step N" values, which is enough to compute positions and allocate KV pages correctly. The only unknown is step N's **sample**, that is, whether the request hit EOS at step N. So a request that hits EOS at step N gets scheduled once more.

**Third, two streams.** The scheduler prepares metadata on its own stream (copying positions, indices and so on asynchronously from pinned memory to the GPU) while the engine computes on another. Before launching, `self.engine.stream.wait_stream(self.stream)` makes the compute wait for the metadata copies; meanwhile the next step's preparation on the scheduler's stream need not wait for the current compute. `ForwardInput` keeps every tensor this step uses, indices included, alive until the results are handled, so none of them is freed and its memory reused while the GPU is still reading it.

## The four problems that follow {#由此带来的四个问题}

"The CPU state is one step ahead" and "a request may be scheduled one step too many" make handling results subtle. The official implementation handles one case (avoiding a double free), and this reimplementation found four more problems, all reproducible in the example and the tests:

@@code examples/ch11_overlap.py@@

@@output ch11_overlap@@

### Problem one: the finished flag arrives one token early {#问题一结束标记提前了一个-token}

Upstream uses `finished = not req.can_decode` to tell whether `max_tokens` has been reached. But when step N is handled, step N+1 has already been launched and `device_len` has already moved on, so `can_decode` reflects the state "after step N+1". A request with `max_tokens=3` is therefore marked finished at its second token (the `(13, True)` above), and the third token comes with another finished message.

The offline interface happens to be unaffected, since it records the token of every message, but an online frontend ends its response at the first `finished=True`, so **the client only ever receives `max_tokens - 1` tokens**. The fix is to use a quantity independent of the scheduling rhythm, namely how many tokens the CPU has actually received:

```python
finished = len(req.input_ids) >= req.max_device_len
```

### Problem two: one stale message after EOS {#问题二eos-之后多发一条过期消息}

A request that hits EOS at step N is scheduled once more (prerequisite two). Upstream records the requests that finished last step in `finished_reqs` and skips **releasing their resources** when handling step N+1, but still builds a reply message for them, the `(6722, False)` above. The frontend has already deleted that request and ignores the message; the detokenizer, however, creates a fresh decoding state for it that never gets an end, which leaks a little memory. The fix is to skip these requests entirely at the top of the loop.

@@code python/minisgl/scheduler/scheduler.py:Scheduler._process_last_data@@

### Problem three: the request slot is reused too early {#问题三请求槽过早复用}

@@diagram overlap-hazard what happens after a request hits EOS at step N@@

A request hits EOS at step N, and when step N's results are handled its resources are released, including the row it holds in the page table / token pool. But step N+1 is running on the GPU right then and still contains that request: it will write its sample into that row of the token pool on the engine stream. If the next scheduling step hands the row to a new request immediately, the write on the scheduler stream (the new request's prompt) has no ordering guarantee against the old write on the engine stream, and the new request's input can be overwritten.

The KV pages have no such problem, because the batch that writes the new request's KV is queued on the engine stream after step N+1. The request slot is different because it is written by the **scheduler stream**. The fix: in overlap mode a released request slot is only really returned at "the next time results are handled", by which point `copy_done.synchronize()` has guaranteed the in-flight batch is done. With no batch in flight (`last_data is None`) it is returned at once, or an interactive setup with only one request slot would stall.

@@code python/minisgl/scheduler/scheduler.py:Scheduler._free_req_resources@@

### Problem four: an abort while prefill is in flight {#问题四prefill-在途时收到-abort}

A disconnecting client triggers an abort. If the abort arrives just after this request's prefill was launched but before its results were handled, upstream's sequence is: the abort releases the request (`cache_req(finished=True)`, so the prompt's KV enters the radix cache); then the prefill results are handled, the request has not finished, and `cache_req(finished=False)` is called once more, on a request that has already been released. The second insertion finds the prefix already in the cache and frees "the duplicate copy", which happens to be the cache's own copy: **the same pages end up both in the free list and in the cache tree**. They may then be handed to a new request and overwrite the cached KV, and the integrity check while the scheduler is idle raises as well (under the radix cache it reproduces as `free_pages(1024) + cache_pages(5) != num_pages(1024)`).

The fix is one line: add the request to `finished_reqs` on abort, and problem two's logic discards the in-flight batch's results for it wholesale.

!!! diff "Differences from upstream"
    All four of these fix the official overlap scheduling, and each has a test: put the official version back and the test fails. Reproducing them on a CPU relies only on the ordering "step N+1 is already launched when step N is handled", not on real GPU concurrency; problem three's test checks the invariant directly, that a request slot being allocated does not belong to any request in an in-flight batch.

!!! upstream "The official implementation"
    - @@upstream scheduler/scheduler.py:Scheduler.overlap_loop@@
    - @@upstream scheduler/scheduler.py:Scheduler._process_last_data@@ (`finished = not req.can_decode`, `if finished and req not in self.finished_reqs`)
    - @@upstream scheduler/scheduler.py:Scheduler._process_one_msg@@ (the abort branch)
    - turning overlap scheduling off: the environment variable `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`, which the ablation in the official README uses

## What it means on a CPU {#在-cpu-上意味着什么}

There is no asynchronous execution on a CPU: `NullStream` and `NullEvent` are no-ops and the computation is already done by the time it is "launched". So overlap scheduling is no faster on a CPU, but **the scheduling logic is identical**, and the four problems above still appear and are still caught by the tests. The performance gain has to be measured on a GPU; the official README suggests an ablation with `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`. The smaller the model and the shorter each step on the GPU, the larger the CPU's share and the more overlap wins. Chapter 21 shows how to run that ablation on a GPU.

## Tests {#测试}

@@code tests/test_ch11_overlap.py:test_overlap_equals_normal_and_hf@@

Both loops across three configurations (radix / naive, chunking, page size 4) produce output identical to Hugging Face. Four more tests cover the four problems above:

@@code tests/test_ch11_overlap.py:test_abort_while_prefill_is_in_flight@@

!!! interview "How to explain it"
    On overlap scheduling: launch step N+1 first, then handle step N's results on the CPU, so the CPU cost of scheduling, batching and detokenization hides behind the GPU's compute. There are three prerequisites: step N+1's input token is written on the GPU by step N's sample (the CPU never needs the value); the request state is advanced at launch time; and the scheduler and the engine use two streams. The price is that the CPU state runs a step ahead of the GPU: a request that sampled EOS at step N may already be in step N+1, so its extra result has to be discarded; the finished check has to use the number of tokens actually received; and a request slot can only be reused once the in-flight batch is done. The gain is largest with small models and small batches, where a GPU step is very short.

## Exercises {#练习}

1. In `overlap_loop`, what do you get if you swap "launch this step" and "handle the previous one" back while keeping the way `last_data` is passed along?
2. The extra step in problem two wastes one computation. Could you know whether step N hit EOS before scheduling step N+1? At what cost?
3. Why does problem three affect only the request slot, while KV pages can be reused at once? If a new request hits the prefix cache, the scheduler stream writes the hit locations into the page table; does that raise a similar problem?

??? success "Answers"
    1. It becomes equivalent to the normal loop (launch, then wait for the result immediately), losing the overlap but staying correct.
    2. You could `synchronize` on step N's result before launching step N+1, but that is exactly the wait overlap scheduling exists to avoid. The other view is to accept the waste: a request computes at most one extra token, so the effect on throughput is proportional to how many requests finish per step and is usually negligible.
    3. The KV pool is only written by kernels on the engine stream, so a new request's KV writes queue behind the old batch; the token-pool row and page-table row of a request slot, however, are written by the scheduler stream (the new request's prompt, the hit locations, the newly allocated pages). On a prefix hit the write goes to the new request's own row, which is fine as long as that row was not just released while an in-flight batch still uses it, which is precisely what deferring the request slot's release guarantees.

## Summary {#小结}

- [x] Overlap scheduling launches this step before handling the previous one, hiding the CPU cost behind the GPU's compute.
- [x] Prerequisites: the input is written on the GPU by the previous step's sample; the request state is advanced at launch time; the scheduler and the engine use two streams.
- [x] The price: the CPU state runs a step ahead, and a request may be scheduled one step too many after EOS.
- [x] This book fixes four problems in the official version: the finished check uses the number of tokens received; stale results are discarded; a request slot is reused only once the in-flight batch is done; and an aborted request joins `finished_reqs`.
