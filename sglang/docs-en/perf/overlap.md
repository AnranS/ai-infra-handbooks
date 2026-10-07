# Overlapped scheduling: hiding the CPU behind the GPU

<p class="lead">v0.4's headline is a "zero-overhead batch scheduler": the scheduler runs one step ahead, so while the GPU computes batch N the CPU is already preparing batch N+1, there is no gap between consecutive decodes in Nsight, and the throughput rises 1.1 times. Its implementation is two classes — <code>Scheduler.event_loop_overlap</code> and a <code>TpModelWorkerClient</code> running on another thread — plus a neat "future token" trick: a token not yet sampled stands in the next batch's input as a negative number, filled in on the GPU by the forward thread. This chapter reads the whole way from the first commit of 16 October 2024 to its becoming the default on 19 November, and every one of its collisions with retraction, constrained decoding and chunked prefill.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. In the non-overlapping loop, what CPU work does one step do? Why does it leave the GPU idle?
    2. What does "one step ahead" mean in the overlapping loop? Batch N+1's `input_ids` are unknown while it is being scheduled; what then?
    3. What is `future_token_ids_map`? Why use negative numbers as placeholders?
    4. What kinds of problem were fixed before the overlapping mode became the default? Which features were incompatible with it for a while?

??? success "Answers for the self-test (answer first, then open this)"
    1. Receiving requests, admission and prefix matching, building the batch's tensors, launching the forward pass, **waiting for it to finish**, copying the sampled results back to the CPU, deciding what has finished, preparing the incremental detokenizing, and sending the output to the detokenizer process. As long as the scheduler only prepares the next batch after waiting for the GPU's result, the GPU has nothing to do during that preparation.
    2. The scheduling thread launches batch N without waiting for the result, and immediately handles batch N−1's result and forms batch N+1. The input token of a decoding request in batch N+1 is exactly the one batch N is about to sample, which is not available at scheduling time, so a "future token id" (a negative index) stands in; once the forward thread has batch N's sampled results it writes them into a mapping table, and before batch N+1's forward pass the negative indices are replaced with the real tokens.
    3. An int32 table on the GPU of length `max_running_requests x 3`, where position k holds the real value of "the k-th future token". The placeholder is `-(k+1)`: real token ids are non-negative so negatives cannot be confused with them, and `resolve_future_token_ids` substitutes them all at once with `torch.where(input_ids < 0, map[-input_ids], input_ids)` without the CPU's involvement.
    4. Races (#1712), illegal CUDA memory accesses (#2048, #2070: a tensor the forward thread was still using freed by the scheduling thread), logprobs (#1795), retraction (#1860), constrained decoding (#2095), chunked prefill's mixed batches (#2158); multimodal models had overlap disabled for a while (#2235), and xgrammar's mask has to wait for the forward thread (#2377).

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/overlap.webp is in Chinese; put it back once the English version exists -->

## A month of commits {#一个月的提交}

```bash title="overlap-commits.sh"
REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | awk '$1 >= "2024-10-15" && $1 <= "2024-12-10"' | grep -iE 'overlap' | cut -c1-96
```

```text title="output"
2024-10-16  dbec2f1847  Launch a thread to overlap CPU and GPU (#1687)
2024-10-18  3db43d1b08  Fix `is_all_ready` for overlap copy (#1710)
2024-10-19  769bf11c05  Fix the race condition in overlap mode (#1712)
2024-10-20  b48edff67f  Split the overlapped version of TpModelWorkerClient into a separate file
2024-10-20  cf470fea32  Make token mapping non-blocking in the overlapped mode (#1740)
2024-10-21  7ce3606891  Faster overlap mode scheduler (#1738)
2024-10-25  e646c5901e  Fix logprob in the overlapped mode (#1795)
2024-10-31  b9fd178f1b  Fix retraction + overlap (#1860)
2024-11-15  29ebe3dff4  fix: align enable_overlap_scheduler naming between code and docs (#2038)
2024-11-15  e5c6715003  Fix core (MI300X) with --enable-overlap (#2048)
2024-11-16  edad373135  Fix illegal memory access in overlap mode & Use more fused triton kernel
2024-11-17  a9e90b4bce  [Minor] Fix styles for overlap mode (#2068)
2024-11-17  116685337e  Fix cuda illegal memory access in overlap mode (#2070)
2024-11-19  ffd20fcd03  Make constrained decoding work for overlap scheduler (#2095)
2024-11-19  7d671e4ad2  Enable overlap by default (#2067)
2024-11-20  722530fa01  Enable overlap scheduler by default for the triton attention backend (#2
2024-11-24  731146f6cb  Fix mixed chunked prefill in overlap mode (#2158)
2024-11-27  fb915bd1a2  Disable overlap scheduler for multimodal models (#2235)
2024-12-06  0e7409adb6  Fix the overlap for xgrammar (#2377)
```

#1687 "Launch a thread to overlap CPU and GPU" of 16 October is the starting point; four days later #1726 split the overlapping version into its own file `tp_worker_overlap_thread.py`; #1738 "Faster overlap mode scheduler" is the version in the blog post; and #2067 made it the default on 19 November. Nearly all of the twenty-odd commits in between are bug fixes — overlapping turns two stretches of sequential code into concurrent ones, and every piece of shared state becomes a problem.

## Two event loops {#两个事件循环}

v0.4.0's `Scheduler` keeps both loops. The ordinary one:

```python title="python/sglang/srt/managers/scheduler.py @ v0.4.0 L376-396" linenums="376"
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            if self.server_args.enable_dp_attention:
                batch = self.prepare_dp_attn_batch(batch)

            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # Self-check and re-init some states when the server is idle
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
```

`run_batch` launches the forward pass and **waits for the result**, `process_batch_result` handles it, and only then does the next round begin. The overlapping one:

```python title="python/sglang/srt/managers/scheduler.py @ v0.4.0 L399-434" linenums="399"
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            if batch:
                result = self.run_batch(batch)
                result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # A dummy first batch to start the pipeline for overlap scheduler.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    self.process_batch_result(tmp_batch, None)

            if self.last_batch:
                tmp_batch, tmp_result = result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # Self-check and re-init some states when the server is idle
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
```

The difference is a few lines: the `result` that `run_batch` returns is no longer the sampled output but a "future", and it goes into `result_queue` with a copy of the batch; this round does not handle it but takes **the previous round's** `(batch, result)` off the queue instead. So the scheduling thread's timeline is: launch batch N → handle batch N−1's result → form batch N+1 → launch batch N+1 → handle batch N's result… and the GPU always has a batch in flight. `DUMMY_FIRST` is there so that the first round has a "previous batch" to handle, and incidentally triggers the sampling information's synchronisation event.

![Figure: overlapped scheduling's timeline and the future tokens](../assets/figures/sgl-overlap-timeline.svg){.aig-svg}

## The forward thread and the future tokens {#前向线程与未来-token}

`TpModelWorkerClient` wraps `TpModelWorker` and runs the forward pass on its own thread and CUDA stream:

```python title="python/sglang/srt/managers/tp_worker_overlap_thread.py @ v0.4.0 L41-46" linenums="41"
def resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )
```

```python title="python/sglang/srt/managers/tp_worker_overlap_thread.py @ v0.4.0 L108-160" linenums="108"
    def forward_thread_func_(self):
        batch_pt = 0
        batch_lists = [None] * 2

        while True:
            model_worker_batch, future_token_ids_ct = self.input_queue.get()
            if not model_worker_batch:
                break

            # Keep a reference of model_worker_batch by storing it into a list.
            # Otherwise, the tensor members of model_worker_batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            batch_lists[batch_pt % 2] = model_worker_batch
            batch_pt += 1

            # Create event
            self.launch_done = threading.Event()
            copy_done = torch.cuda.Event()

            # Resolve future tokens in the input
            input_ids = model_worker_batch.input_ids
            resolve_future_token_ids(input_ids, self.future_token_ids_map)

            # Run forward
            logits_output, next_token_ids = self.worker.forward_batch_generation(
                model_worker_batch, self.launch_done
            )

            # Update the future token ids map
            bs = len(model_worker_batch.seq_lens)
            self.future_token_ids_map[
                future_token_ids_ct + 1 : future_token_ids_ct + bs + 1
            ] = next_token_ids

            # Copy results to the CPU
            if model_worker_batch.return_logprob:
                logits_output.next_token_logprobs = logits_output.next_token_logprobs[
                    torch.arange(len(next_token_ids), device=self.device),
                    next_token_ids,
                ].to("cpu", non_blocking=True)
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                    )
                    logits_output.normalized_prompt_logprobs = (
                        logits_output.normalized_prompt_logprobs.to(
                            "cpu", non_blocking=True
                        )
                    )
            next_token_ids = next_token_ids.to("cpu", non_blocking=True)
            copy_done.record()

            self.output_queue.put((copy_done, logits_output, next_token_ids))
```

When the scheduling thread calls `forward_batch_generation` nothing is forwarded: the batch goes into `input_queue` and a stretch of future token ids is allocated and returned **immediately**:

```python title="python/sglang/srt/managers/tp_worker_overlap_thread.py @ v0.4.0 L181-210" linenums="181"
    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            scaling_penalties=sampling_info.scaling_penalties,
            linear_penalties=sampling_info.linear_penalties,
        )

        # A cuda stream sync here to avoid the cuda illegal memory access error.
        torch.cuda.current_stream().synchronize()

        # Push a new batch to the queue
        self.input_queue.put((model_worker_batch, self.future_token_ids_ct))

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.future_token_ids_ct = (
            self.future_token_ids_ct + bs
        ) % self.future_token_ids_limit
        return None, future_next_token_ids
```

`future_next_token_ids = arange(-(ct+1), -(ct+1+bs), -1)`: each of batch N's bs requests gets a negative number. The scheduling thread writes those negatives into the requests as "the token sampled" and assembles them into batch N+1's `input_ids`; the forward thread, actually running batch N, gets `next_token_ids` and writes them into `future_token_ids_map[ct+1 : ct+bs+1]`; and before batch N+1's forward pass, `resolve_future_token_ids` replaces the negatives in the input with the table's real values. The CPU never touches a token value in all of this — which is the heart of "zero overhead": **the scheduler can form the next batch without knowing what the previous one sampled**. The table's length of `max_running_requests x 3` guarantees that placeholders two or three batches old are not overwritten.

The forward thread also copies the results back to the CPU (`non_blocking=True` plus a `copy_done` event), and the scheduling thread waits on that event before handling batch N's result (`resolve_batch_result`); `batch_lists[batch_pt % 2]` keeps a reference to the two most recent batches so that their tensors are not freed before the forward thread has used them — the comment says this is to avoid a "CUDA illegal memory access", exactly the kind of problem #2048 and #2070 fixed.

## Collisions and fixes {#冲突与修复}

Overlapping defers "the previous batch's result" by a step, so every piece of logic that assumed "the result is known as soon as this step ends" had to be rewritten:

| Feature | The collision | The fix |
| --- | --- | --- |
| Retraction | retracting modifies the batch, but the forward thread may still be using it | #1860: retract while handling the previous batch's result, and copy the batch |
| Logprobs | the result is in the forward thread and the scheduling thread has to wait for the copy | #1795: the logprob copy follows the `copy_done` event |
| Constrained decoding | the mask has to be computed from the FSM's state before sampling, and that state depends on the previous step's sample | #2095, #2377: the mask is filled in the forward thread after the `sampling_info_done` event |
| Chunked prefill's mixed batches | the decoding requests in a mixed batch use future tokens too | #2158 |
| Multimodal | the image preprocessing and `pad_input_ids` are in the scheduling thread | #2235 disabled it first, with support added gradually |

It became the default on 19 November (#2067), and the Triton backend the next day (#2105). The v0.4 blog's number is 1.1 times, with an Nsight screenshot attached: no gaps on the GPU between consecutive decodes.

## Design trade-offs {#设计取舍}

- **A thread rather than a process.** The forward thread and the scheduling thread share the CUDA context and the tensors, so passing a reference to a `ModelWorkerBatch` is enough; the price is the GIL and races over shared state, and the bug list above is the bill.
- **Future tokens rather than waiting.** The alternative is to have the scheduling thread wait for the sampled result before forming the next batch (overlapping only the result handling); SGLang chose to stand in even for the input, overlapping more thoroughly, at the price of a mapping table and a "resolve" wherever tokens are read.
- **Both loops kept.** The ordinary loop stays as a way out for debugging and for incompatible features.

## What happened afterwards {#后来怎么样了}

```bash title="overlap-files.sh"
REF=${REF:-29f6d408c0}
for t in v0.4.0 v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s tp_worker_overlap_thread.py：%4d 行   scheduler.py：%5d 行\n' "$t" "$(git show "$t:python/sglang/srt/managers/tp_worker_overlap_thread.py" 2>/dev/null | wc -l)" "$(git show "$t:python/sglang/srt/managers/scheduler.py" | wc -l)"
done
```

```text title="output"
v0.4.0      tp_worker_overlap_thread.py： 231 行   scheduler.py： 1500 行
v0.4.6      tp_worker_overlap_thread.py： 240 行   scheduler.py： 2041 行
v0.5.0rc0   tp_worker_overlap_thread.py： 296 行   scheduler.py： 2589 行
29f6d408c0  tp_worker_overlap_thread.py：   0 行   scheduler.py： 5969 行
```

- The roadmap of 2025-10 (issue #7736) made "overlap scheduler simplification" a priority, and after #11762 the forward thread's logic was reorganised, with `scheduler.py`'s two loops still there.
- Speculative decoding's v2 (H2 2025) brought drafting and verification into the overlapping flow too, and `future_token_ids_map`'s idea was generalised to several tokens.
- PD disaggregation's decode side and DP attention's idle batch each added their own branch in the overlapping loop.

## Exercises {#练习}

**1. The mapping table's length.** Why `max_running_requests x 3` rather than `x 2`? When would `x 2` go wrong?

??? success "A way to approach it"
    The scheduling thread may have issued batches N+1 and N+2 before the forward thread has written batch N's results (two batches backed up in the queue), and with the batch being resolved that is three batches' placeholders valid at once; at `x 2`, batch N+2's placeholders could overwrite batch N's values before they are resolved.

**2. Verify one collision yourself.** Read #1860's diff (`git show b9fd178f1b`) and explain why retraction needs a `batch.copy()`, and what happens to a retracted request's future token.

??? success "A way to approach it"
    Retraction happens while handling batch N's result, and batch N+1 has already been issued; modifying the batch directly would affect the object the forward thread is using, so the modification is made on a copy. A retracted request's placeholder token is no longer referenced by any batch, so the table's value simply falls out of use.

**3. What turning overlap off costs.** Find `--disable-overlap-schedule`'s handling at the baseline commit and list the modes under which it is turned off automatically.

??? success "A way to approach it"
    `git grep -n 'disable_overlap_schedule' 29f6d408c0 -- python/sglang/srt/server_args.py` shows the conditions for the automatic shutdown (certain attention backends, certain speculative-decoding configurations, particular hardware).

!!! interview "How to answer in an interview"
    "How does an inference engine overlap the CPU and the GPU?" — Give SGLang's three pieces: the scheduling thread runs one step ahead (the result queue defers handling by a round), the forward thread has its own CUDA stream, and future tokens stand in as negative numbers resolved on the GPU. Then the price: every piece of logic assuming "the result is known" (retraction, constrained decoding, logprobs, mixed batches) had to change, and a month went into fixing races before it became the default. Comparing it with vLLM V1's asynchronous scheduling (`AsyncScheduler`) shows the two ideas are close and the implementations differ.

## Summary {#小结}

- [x] It started with #1687 on 2024-10-16 and became the default with #2067 on 11-19; the v0.4 blog: 1.1 times, with no gaps on the GPU.
- [x] The implementation: `event_loop_overlap` defers result handling by a round, plus `TpModelWorkerClient`'s forward thread, plus future tokens standing in as negatives and resolved on the GPU.
- [x] The price is the races concurrency brings and a feature-by-feature reconciliation with retraction, constrained decoding, mixed batches and multimodal input; both loops are still there today.
