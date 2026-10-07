# The scheduler skeleton and the offline interface

<p class="lead">In the last chapter we played the scheduler by hand: allocating KV locations, filling in the batch fields, appending tokens. This chapter automates it. mini-sglang's scheduler is four small managers, for the request table, the KV cache, the decode set and the prefill queue, and the main loop is barely a dozen lines: receive, pick a batch, prepare, forward, handle the results. We start with the plainest version, with no prefix reuse, no chunking and no overlap, and drive it from the offline <code>LLM</code> interface.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is the difference between continuous batching and static batching?
    2. Does a mini-sglang step do only prefill or only decode, or does it mix them? Which comes first?
    3. The decode set is a `set`. Why sort by `uid` when assembling a batch?
    4. The offline `LLM` interface has no ZMQ. How does it reuse the scheduler's main loop?

??? success "Answers (try it yourself first, then expand)"
    1. Static batching waits for a whole batch of requests to finish before starting the next; continuous batching reassembles the batch every step, so a finished request leaves at once and a new one joins at any time.
    2. One kind per step: if there is a prefill that can be admitted it goes first, otherwise it decodes.
    3. Under tensor parallelism every rank runs its own scheduler and they must assemble exactly the same batch, because the order of requests decides the KV layout. A `set` iterates in an unspecified order, so sorting by `uid` keeps the ranks in agreement.
    4. `LLM` subclasses the scheduler and overrides the two methods for receiving messages and sending results: it takes inputs from an in-memory request list and collects results locally, reusing the main loop unchanged.

**Files you will write**: `scheduler/config.py`, `scheduler/utils.py`, `scheduler/decode.py`, `scheduler/prefill.py`, `scheduler/scheduler.py`, `llm/llm.py` (`scheduler/table.py` came in the last chapter; `scheduler/cache.py` comes in the next one, and this chapter uses its naive mode).

@@video batching Animation: continuous batching and admission control (about 1.5 minutes, covering this chapter and the next; Chinese narration and subtitles)@@

## The parts {#组成}

@@code python/minisgl/scheduler/scheduler.py:Scheduler.__init__@@

@@diagram scheduler-loop what the scheduler is made of, and its main loop@@

| Manager | Responsibility | Chapter |
| --- | --- | --- |
| `TableManager` | rows of the page table / token pool: one per running request | [KV pool](../compute/kvcache.md) |
| `CacheManager` | page allocation and release in the KV pool, the face of the prefix cache | [CacheManager](cache-manager.md) |
| `DecodeManager` | the set of requests currently decoding | this chapter |
| `PrefillManager` | the queue waiting for prefill, and assembling prefill batches | this chapter, chunking in [chapter 10](chunked-prefill.md) |

The scheduler also inherits `SchedulerIOMixin`, which provides `receive_msg` and `send_result`; in an online service they go over ZMQ (chapter 14), and offline `LLM` overrides them.

## The main loop {#主循环}

@@code python/minisgl/scheduler/scheduler.py:Scheduler.normal_loop@@

Four steps per iteration:

1. **Receive.** If there is nothing to do at all (nothing waiting for prefill, nothing decoding), block and wait; otherwise take only the messages that have already arrived, without waiting.
2. **Pick a batch.** Prefill if anything is waiting, decode otherwise:

    @@code python/minisgl/scheduler/scheduler.py:Scheduler._schedule_next_batch@@

    This is the "prefill first" policy: new requests get their first token as soon as possible (low TTFT), at the cost of interrupting the decoding requests for a step. One kind per step, never mixed, which keeps each step's batch shape simple (a decode batch can use CUDA Graph) and is one of the places mini-sglang is simpler than production SGLang.
3. **Prepare and run forward.** `_prepare_batch` fills in the batch's fields and `_forward` calls the engine.
4. **Handle the results.** Give the new token to the request, decide whether it is finished, reply to the detokenizer, and release the resources of whatever finished.

`run_forever` just calls this loop over and over (the overlap version, `overlap_loop`, arrives in chapter 11).

## Two of the managers {#两个管理器}

`PrefillManager` keeps a first-come-first-served waiting queue and uses a `PrefillAdder` each step to admit as many requests as it can, starting from the head:

@@code python/minisgl/scheduler/prefill.py:PrefillManager.schedule_next_batch@@

`PrefillAdder` does three things per request: match the cached part in the prefix cache, check that there is a free request slot and enough KV space (admission control, covered next chapter), and write this step's tokens into the token pool. It stops at the first request that does not fit, rather than skipping it to admit smaller ones behind it, which keeps things fair and stops any request from being jumped over forever.

`DecodeManager` is simpler still, just a set of requests:

@@code python/minisgl/scheduler/decode.py:DecodeManager@@

After each forward pass, `filter_reqs` merges this step's requests in and drops the ones that can no longer generate (`can_decode` false). A request that has just prefilled joins the decode set this way, naturally. Batches sort by `uid`: under tensor parallelism every rank has its own `DecodeManager`, and Python's `set` iterates in an order that depends on object addresses, which can differ between processes; yet every rank's batch must be identical, the same requests in the same order, or an all-reduce adds different requests' data together. The most recent commit in the official repository fixes exactly this.

## Preparing a batch {#准备-batch}

@@code python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch@@

In order:

1. **Pad** (chapter 18; a no-op without CUDA Graph);
2. **Allocate KV pages**: find locations for this step's tokens and write them into the page table (chapter 8);
3. **Positions**: `[cached_len, device_len)` per request, packed into one dimension;
4. **Two pairs of two-dimensional indices**: `input_tuple = (row, position)` to take the input from the token pool and `out_loc` from the page table; `write_tuple = (row, device_len)` to write the sample back into the token pool's next position;
5. **Attention metadata** and **sampling parameters**.

@@code python/minisgl/scheduler/scheduler.py:_make_write_tuple@@

Then the forward pass:

@@code python/minisgl/scheduler/scheduler.py:Scheduler._forward@@

Both the input `token_pool[input_tuple]` and the write-back `token_pool[write_tuple] = next_tokens` are indexing operations on the device, so the CPU never needs to know the token values.

## Handling the results {#处理结果}

@@code python/minisgl/scheduler/scheduler.py:Scheduler._process_last_data@@

For each request in the batch: append the new token to the CPU-side `input_ids`, decide whether it has finished (it reached `max_tokens`, or hit EOS without `ignore_eos`), and build one `DetokenizeMsg`. A finished request releases its resources: it returns its page-table row and hands its KV to the cache (the naive cache frees all of it). A request that has just prefilled hands the prompt's KV to the prefix cache (chapter 9).

Two things in this code are there for overlap scheduling in chapter 11, namely `finished_reqs` and using `len(req.input_ids)` rather than `req.can_decode` to judge the length, and that chapter explains them.

Handling a new request:

@@code python/minisgl/scheduler/scheduler.py:Scheduler._process_one_msg@@

When the prompt plus `max_tokens` exceeds the model's maximum length, `max_tokens` is cut back; a request whose prompt alone is too long is dropped outright (upstream likewise only logs a warning, and the frontend never gets a reply, which is one thing worth improving).

## The offline LLM interface {#离线接口-llm}

@@code python/minisgl/llm/llm.py:LLM@@

`LLM` subclasses `Scheduler`, passes `offline_mode=True` so the IO layer creates no ZMQ queues, and overrides two methods:

- `offline_receive_msg`: take requests from the pending list and feed them in batches within the prefill budget (`max_extend_tokens`). When the scheduler has nothing to do and would block waiting for a message while the list is also empty, everything is done and it raises `RequestAllFinished` to break out of `run_forever`;
- `offline_send_result`: record each reply's token on the matching request (EOS is not recorded).

This small trick lets the offline interface share the main loop with the online service, so testing one tests the other's scheduling logic too.

## Running it {#运行}

@@code examples/ch07_scheduler.py@@

@@output ch07_scheduler@@

The first step prefills all five requests together (24 tokens packed into one dimension). uid3, with `max_tokens=1`, finishes right after prefill and never reaches decode; uid1 leaves after one decode step; the rest carry on and the batch shrinks. That is continuous batching: requests join and leave at their own pace, without waiting for the whole batch. All five outputs match what Hugging Face generates for them individually.

!!! upstream "The official implementation"
    - the main loop: @@upstream scheduler/scheduler.py:Scheduler.normal_loop@@
    - preparing and running forward: @@upstream scheduler/scheduler.py:Scheduler._prepare_batch@@, @@upstream scheduler/scheduler.py:Scheduler._forward@@
    - the prefill queue: @@upstream scheduler/prefill.py:PrefillManager.schedule_next_batch@@
    - the decode set: @@upstream scheduler/decode.py:DecodeManager@@
    - the offline interface: @@upstream llm/llm.py:LLM@@

## Tests {#测试}

@@code tests/test_ch07_scheduler.py:test_requests_with_different_lengths_leave_and_join@@

!!! interview "Answering in an interview"
    On the scheduler: continuous batching reassembles the batch step by step, so requests join at any time and leave the moment they are done, without waiting for the whole batch (in static batching a short request has to sit through the longest one). mini-sglang's main loop is receive, pick a batch, prepare, forward, handle results, and it does one kind per step: prefill first (first come first served, stopping at the first that does not fit), decode otherwise. The decode set sorts by uid when assembling, so every TP rank builds exactly the same batch. Against vLLM V1: vLLM mixes prefill and decode in one step under a unified token budget, while mini-sglang keeps them apart, which is simpler to implement but lets a long prefill stall decode for a step. The offline `LLM` interface replaces only the receive and send methods and reuses the same main loop.

## Exercises {#练习}

1. Change the policy to "decode first": decode whenever anything is decoding, and prefill only when nothing is. What goes wrong? (Hint: think about when a new request gets served in a system that always has something decoding.)
2. `PrefillManager` stops at the first request that does not fit. What happens to throughput if it skips that one and keeps trying the ones behind it? What is the risk?
3. In `_process_one_msg` an over-long prompt is dropped and the frontend never hears back. How would you return an error instead? Which message types need changing?

??? success "Answers"
    1. As long as the decode set is non-empty, a new request never gets prefilled, so TTFT is unbounded (starvation). A practical compromise is to cap how many decode steps may run in a row, or to mix prefill and decode in one batch the way production SGLang does.
    2. Throughput may go up, since small requests fill the budget, but a large request can be jumped over by later small ones forever and starve. That needs an aging mechanism, where waiting longer raises priority.
    3. Have the scheduler reply with a `DetokenizeMsg` carrying an error flag (an added `error` field with `finished=True`), have the detokenizer turn it into a `UserReply` with the error message, and have the API server return a 400 on that basis.

## Summary {#小结}

- [x] The scheduler is four managers plus a main loop: receive, pick a batch (prefill first, one kind per step), prepare, forward, handle results.
- [x] The prefill queue is first come first served and stops at the first request that does not fit; the decode set is updated after each forward pass and sorts by uid so every rank agrees.
- [x] Inputs are taken from the token pool and samples written back to it, all on the device.
- [x] The offline `LLM` interface overrides the two IO methods and reuses the same main loop.
