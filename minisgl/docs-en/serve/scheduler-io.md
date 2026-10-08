# The scheduler's IO and multi-rank synchronization

<p class="lead">The scheduler's main loop calls only two IO methods, <code>receive_msg(blocking)</code> and <code>send_result(reply)</code>. Behind them is <code>SchedulerIOMixin</code>: on one GPU it talks ZMQ directly, and under tensor parallelism only rank 0 connects to the tokenizer and every rank has to see exactly the same messages each step. This chapter implements it and explains why "exactly the same" matters so much.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Under tensor parallelism, why does every rank run a full scheduler instead of rank 0 scheduling and the others following orders?
    2. The number of messages rank 0 receives varies per step. How do the other ranks know how many to take?
    3. Who sends the results to the detokenizer?
    4. When is `receive_msg(blocking=True)` called? What happens before it blocks?

??? success "Answers (try it yourself first, then expand)"
    1. Each rank's KV pool and page table live on its own GPU, and the scheduling decisions determine the KV layout directly. Running a full scheduler on every rank means that, given the same input messages, deterministic scheduling produces the same batch and the same KV layout everywhere, with no need to send the scheduling result (a lot of metadata) to the other ranks each step.
    2. rank 0 uses one broadcast to tell the other ranks how many messages this step has, and they take that many from PUB / SUB.
    3. Only rank 0: every rank samples the same token anyway, so one copy is all that is needed.
    4. Only when there is no work at all (no running requests, nothing pending) does it block for a new message, so it does not spin; before blocking it finishes handling every in-flight result and runs the idle-time checks.

**Files you will write**: `scheduler/io.py`.

## Why every rank schedules {#为什么每个-rank-都调度}

Under tensor parallelism each rank holds part of the model and part of the KV cache (split by KV head, chapter 16). Every step, all ranks must compute **the same batch**: the same requests, in the same order, writing the same tokens to the same locations in the KV pool. Otherwise an all-reduce adds different requests' partial sums together.

One option is to have rank 0 schedule and broadcast each step's batch (the request list, positions, page allocations and so on) to the others. mini-sglang's way uses less communication: **every rank runs a full scheduler, and as long as they see exactly the same input messages, deterministic scheduling logic reaches exactly the same decisions on every rank**, keeping the KV pool, the page table and the radix cache in agreement throughout. The only thing to broadcast is the raw messages, a few hundred bytes each.

That requires the scheduling logic to be deterministic, which is why `DecodeManager` sorts by `uid` when assembling a batch in chapter 7, to remove any difference in `set` iteration order between processes.

## Receiving {#收消息}

@@code python/minisgl/scheduler/io.py:SchedulerIOMixin.__init__@@

The constructor picks the receive and send methods by whether this is rank 0 and whether there are other ranks. Offline mode swaps in the two methods `LLM` provides (chapter 7).

The single-GPU version:

@@code python/minisgl/scheduler/io.py:SchedulerIOMixin._recv_msg_single_rank@@

With nothing to do (`blocking=True`) it first calls `run_when_idle` (one memory integrity check, chapter 8), then blocks for the first message; after that it takes everything that has already arrived without waiting further.

Rank 0 with several GPUs:

@@code python/minisgl/scheduler/io.py:SchedulerIOMixin._recv_msg_multi_rank0@@

1. the first message received while blocking is forwarded verbatim right away;
2. everything that has already arrived is taken (only rank 0 knows how many that is);
3. **one broadcast tells the other ranks how many more there are this step**;
4. those messages are forwarded verbatim and decoded locally.

The other ranks:

@@code python/minisgl/scheduler/io.py:SchedulerIOMixin._recv_msg_multi_rank1@@

While blocking they first take one message from PUB/SUB (matching rank 0's first), then join the broadcast to learn the count and take that many. The `blocking` argument always agrees on both sides, because it depends on the scheduler's state and every rank's state is identical.

The count is broadcast over the CPU gloo process group (`tp_cpu_group`). On a GPU the tensors go over NCCL and the control information over a separate gloo group, out of each other's way.

## What the two ranks received {#看两个-rank-收到了什么}

Without loading a model, using only the scheduler's IO: the main process plays the tokenizer and sends 3 messages in a row, and two processes play the two ranks.

@@code examples/ch14_scheduler_io.py@@

@@output ch14_scheduler_io@@

rank 0 blocks and receives message 1 on the first step, then takes whatever else had arrived by then; rank 1 learns the count from the broadcast and receives the same messages from PUB/SUB. However the messages arrive and in whatever batches, both ranks see the same sequence. (This demo hung the first time it ran, because of PUB/SUB's slow-joiner problem; see "Difference from upstream: XPUB" in the last chapter.)

## Sending results {#发结果}

@@code python/minisgl/scheduler/io.py:SchedulerIOMixin._reply_tokenizer_rank0@@

Only rank 0 sends results to the detokenizer, and `send_result` does nothing on the other ranks. Every rank samples the same thing anyway: after the all-gather every rank has the complete LM head logits, so greedy decoding agrees by construction, and with random sampling every rank uses the same seed (`torch.manual_seed(42)` in `Engine`) and the same call sequence, so the results agree too.

!!! upstream "The official implementation"
    - @@upstream scheduler/io.py:SchedulerIOMixin@@
    - the most recent official commit (#113) fixes exactly the case where the decode batches on different ranks ordered their requests differently, which violates the "exactly the same" principle head-on.

## Tests {#测试}

Multi-rank IO is covered by the end-to-end tensor-parallel tests of chapter 16: TP=2 and TP=4 services produce output identical to single-GPU Hugging Face.

!!! interview "How to explain it"
    On multi-rank synchronization: under tensor parallelism every rank runs a full scheduler, and as long as all ranks receive exactly the same messages, deterministic scheduling assembles the same batch and allocates the same KV locations, so rank 0 never has to send its scheduling results down each step. The mechanism is that rank 0 receives messages, forwards the raw bytes verbatim, and uses one broadcast to tell the others how many there are this step, which they then take from PUB/SUB; only rank 0 sends results to the detokenizer, since every rank sampled the same thing. It blocks for a message only when there is nothing to run, and finishes the in-flight batch before blocking. The prerequisite for determinism is that scheduling never depends on anything that differs between ranks, such as the clock or a set's iteration order, which is why batches sort by uid.

## Exercises {#练习}

1. With random sampling, what happens if the ranks' random number generators are in different states? Only rank 0 sends results, so it looks fine. Is it?
2. What goes wrong if the count broadcast is removed and rank 1 decides for itself with `empty()` whether more messages remain?
3. Why does rank 0 "forward first, decode second" rather than "decode first, re-encode and forward"?

??? success "Answers"
    1. The ranks sample different tokens, and each rank writes its own token pool, so the next step's inputs differ, an all-reduce adds data from different sequences together, and every rank's computation is wrong (rank 0's output merely still looks like *some* sequence). So every rank must use the same seed and make the same number of random calls in the same order.
    2. PUB/SUB messages reach rank 1 with a delay, so when rank 1 decides "no more messages", rank 0 may already have forwarded several that are still in flight. The two sides then handle different sets of messages this step and their scheduling decisions diverge.
    3. It saves an encode, and it guarantees that what is forwarded is byte-for-byte identical.

## Summary {#小结}

- [x] Every rank runs a full scheduler, and given the same input messages, deterministic scheduling produces the same batch and the same KV layout.
- [x] rank 0 forwards messages verbatim and uses one broadcast to tell the other ranks how many there are this step; they take that many from PUB/SUB.
- [x] Only rank 0 replies to the detokenizer, since every rank sampled the same thing.
