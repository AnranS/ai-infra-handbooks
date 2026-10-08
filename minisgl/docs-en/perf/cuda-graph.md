# CUDA Graph

<p class="lead">One decode step of a 28-layer model launches hundreds of kernels. At a small batch size each kernel computes for only a few microseconds, while the CPU takes a few microseconds to launch one, so the GPU spends most of its time waiting for the CPU. CUDA Graph records a whole forward pass so one replay launches every kernel. The price is that every address the recorded kernels read and write is now fixed, so every input has to be copied into fixed buffers first. This chapter implements mini-sglang's <code>GraphRunner</code> and verifies the "every input goes through a fixed buffer" contract with a CPU emulation.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why is CUDA Graph used for decode and not for prefill?
    2. The recorded batch size is 4 and the real batch has 3 requests. What now? Where does the padded request's output go?
    3. The attention metadata (each request's KV length, the page table) differs every step. How does the graph see the new values?
    4. Why is the largest batch size recorded first?

??? success "Answers (try it yourself first, then expand)"
    1. A decode step has very few tokens but hundreds of small kernels, so the CPU's launch cost dominates and a graph removes it; a prefill's shape (its token count) differs every time across a wide range, which is impossible to record for, and prefill does enough compute that launch overhead never mattered.
    2. Pad the batch to 4 with a dummy request: its inputs and metadata live in the fixed buffers, its KV is written to the extra page allocated for it in the KV pool, and its result is discarded.
    3. All the metadata lives in buffers fixed at capture time, which are the only addresses the graph knows; before each replay, this step's new values are copied into those buffers.
    4. Recording the largest first makes it allocate temporary memory from the pool at its largest requirement, after which the smaller graphs reuse the same memory pool and add nothing.

**Files you will write**: `engine/graph.py`, plus `init_capture_graph`, `prepare_for_capture` and `prepare_for_replay` in the attention backends.

@@video cudagraph Animation: capturing, padding and replaying a CUDA Graph (about 1.5 minutes, Chinese narration and subtitles)@@

## Capture and replay {#录制与-replay}

A CUDA Graph records "kernels plus arguments", and the pointers among those arguments are fixed at capture time. So:

1. **inputs live in fixed buffers**. Before capture, the batch's `input_ids`, `positions` and `out_loc` point at slices of the buffers; before replay, the current batch's values are copied in;
2. **the attention metadata lives in fixed buffers too**, which the attention backend takes care of: at capture the metadata points at the backend's own buffers, and before replay the new metadata is copied in;
3. **one graph per batch size**, with a real batch padded up to the nearest recorded size.

@@diagram cuda-graph capture and replay: every input goes through a fixed buffer@@

@@code python/minisgl/engine/graph.py:GraphCaptureBuffer@@

@@code python/minisgl/engine/graph.py:GraphRunner._capture_graphs@@

The batch used for capture is all dummy requests (chapter 6: a dummy has its own row in the page table, pointing at its own page in the KV pool). Each batch size runs once normally first (a warm-up that triggers the libraries' lazy initialization so it does not get recorded into the graph), then runs again inside a `torch.cuda.graph` context to capture. Recording starts from the largest batch size, and the smaller graphs then reuse the same memory pool (`pool`), so memory use is set by the largest.

The batch sizes recorded by default are 1, 2, 4 and every multiple of 8 from 8 up to a ceiling set by memory (256 on cards above 80 GB, 160 otherwise).

## Padding and replay {#补齐与-replay}

@@code python/minisgl/engine/graph.py:GraphRunner.pad_batch@@

`pad_batch` is the first thing the scheduler calls when preparing a batch (chapter 7): when a graph can be used, dummy requests pad the batch up to the nearest recorded size. The padded dummies get positions, inputs and metadata too, their KV goes into the dummy page, and their logits are discarded (`logits[:batch.size]`).

@@code python/minisgl/engine/graph.py:GraphRunner.replay@@

The attention backend's three hooks, with the reference backend as the example (the same as the official FlashAttention backend's approach):

@@code python/minisgl/attention/torch_backend.py:TorchAttnBackend.prepare_for_capture@@

@@code python/minisgl/attention/torch_backend.py:TorchAttnBackend.prepare_for_replay@@

The FlashInfer backend works differently: it offers wrappers made for CUDA Graph that take fixed-address buffers at construction, and `plan` copies the new metadata into them, so planning again before replay is all it takes (`FlashInferBackend.prepare_for_replay`, chapter 17).

## The CPU emulation {#cpu-上的仿真}

There is no CUDA Graph on a CPU, but we still want to check that the "every input goes through a fixed buffer" contract holds, because that is the most common source of CUDA Graph bugs: one input is not copied in, the graph reads the old value from capture time (or from the previous step), and nothing reports an error.

@@code python/minisgl/engine/graph.py:EmulatedGraph@@

During "capture" `EmulatedGraph` records the function to execute and the batch as it was at capture time. In that batch the input fields point at the buffers and the attention metadata points at the backend's buffers; at replay it runs the forward pass again with *that* batch rather than the current one. So it behaves like a real CUDA Graph: **it can only see what is in the buffers**.

@@code examples/ch18_cuda_graph.py@@

@@output ch18_cuda_graph@@

Three requests decode with every step padded to 4 and replayed, and the output is identical to running without a graph. Then a bug is introduced on purpose: `positions` is not copied before replay. RoPE inside the graph reads the positions left in the buffer from capture time (all zeros) and the output is wrong from the second token on, with no error anywhere. It would look exactly the same on a real GPU.

!!! diff "Difference from upstream"
    `EmulatedGraph` is ours: on a CPU with `cuda_graph_max_bs` set explicitly, it stands in for `torch.cuda.CUDAGraph` (graphs are off by default on a CPU). The capture loop, the padding and the replay logic are the same as upstream's.

!!! upstream "The official implementation"
    - @@upstream engine/graph.py:GraphRunner@@
    - choosing the batch sizes: @@upstream engine/graph.py:_determine_cuda_graph_bs@@
    - the attention backends' hooks: @@upstream attention/fa.py:FlashAttentionBackend.prepare_for_replay@@, @@upstream attention/fi.py:FlashInferBackend.prepare_for_capture@@
    - Note: `--cuda-graph-max-bs 0` turns CUDA Graph off, and graphs must be destroyed before the NCCL resources are released or it can hang (per the comment on upstream's `destroy_cuda_graphs`).

## Tests {#测试}

@@code tests/test_ch18_cuda_graph.py:test_forgetting_to_copy_an_input_breaks_replay@@

The same file also has end-to-end tests for the reference, FlashInfer and FlashAttention backends under the emulated graph (every decode step replays, and the output matches Hugging Face), plus the batch-size list computation.

!!! interview "How to explain it"
    On CUDA Graph: a decode step is hundreds of small kernels where the CPU's launch cost exceeds the GPU's compute, so CUDA Graph records a whole decode forward pass and replays it in one go; a prefill's token count varies widely and it is compute-heavy anyway, so it neither needs nor suits recording. What is recorded are fixed addresses, so every input and all the attention metadata must live in fixed buffers with the new values copied in before replay (miss one and it fails silently). One graph is recorded per batch size and a real batch is padded with dummy requests to the nearest one, with the dummies' KV written to a page reserved for them. The largest batch size is recorded first so the memory pool it allocates is reused by the later graphs. The gain is largest with small models and small batches.

## Exercises {#练习}

1. With 5 requests in the batch and `[1, 2, 4, 8]` recorded, what is it padded to? How much extra compute does that cost? How would you choose the list of batch sizes to balance "number of graphs (memory)" against "padding waste"?
2. Could sampling be recorded into the graph too? Why does mini-sglang not do so?
3. Under tensor parallelism, can the all-reduce inside the graph be recorded? What has to be watched?

??? success "Answers"
    1. Padded to 8, computing 3 extra dummy requests, about 60% more. At small batch sizes every size is common, so the steps should be small (1, 2, 4, 8); at large batch sizes the kernels are already well filled and the padding waste is a smaller fraction, so the steps can be wide (every 8 or 16). That is the thinking behind the official default list.
    2. It could, but the sampling parameters (temperature, top-k, top-p) differ per request and the random-number state is needed too, all of which would have to go into fixed buffers; and greedy sampling is just one argmax, which costs almost nothing.
    3. Yes, NCCL supports capturing collectives in a CUDA Graph. Every rank has to capture and replay the same graphs in the same order (chapter 14's "every rank identical" again), and the graphs must be destroyed before the NCCL communicator.

## Summary {#小结}

- [x] CUDA Graph records a whole decode forward pass so one replay launches every kernel, removing the CPU's launch cost at small batch sizes.
- [x] Every input and all the attention metadata must live in fixed buffers and be copied in before replay; there is one graph per batch size and a real batch is padded with dummy requests.
- [x] `EmulatedGraph` emulates the "can only see the buffers" behaviour on a CPU: miss one input copy and the output is wrong, silently.
