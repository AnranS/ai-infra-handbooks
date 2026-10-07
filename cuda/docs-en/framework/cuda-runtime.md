# PyTorch's CUDA runtime: asynchrony, streams and memory

<p class="lead">On a GPU, every PyTorch operator call merely <strong>queues</strong> a kernel and returns, so the Python code and the GPU run forward in parallel. An inference engine's performance depends largely on keeping that asynchrony: one careless synchronization makes the CPU wait for the GPU and then the GPU wait for the CPU. This chapter covers PyTorch's runtime layer over CUDA: asynchronous execution and synchronization points, streams, the caching allocator and CUDA Graphs.</p>

!!! note "This chapter's code needs an NVIDIA GPU"
    The scripts here are only syntax-checked in the book's verification environment (which has no GPU); they run directly on WSL2 or Linux with an NVIDIA GPU. The concepts match the CUDA handbook's [streams, concurrency and CUDA Graphs](../tools/streams.md), which covers the same mechanisms in CUDA C++.

!!! question "Self-test: if you can answer these, skip the chapter"
    1. When the line `y = x @ w` finishes on a GPU, has `y` been computed?
    2. Which common operations implicitly synchronize the CPU and the GPU?
    3. How do you measure a stretch of GPU code correctly?
    4. Why call `record_stream` when using a tensor on another stream?
    5. How do `torch.cuda.memory_allocated()` and `memory_reserved()` differ?

??? success "Answers (try it yourself first, then expand)"
    1. No: the line only submits the matrix multiply to the CUDA stream and returns, and `y` is a tensor that "will be computed"; reading its values (`.item()`, `.cpu()`) is what waits for the GPU.
    2. `.item()`, `.cpu()`, `.tolist()`, printing a tensor, data-dependent control flow (`if x.sum() > 0`), data-dependent shapes (`nonzero`, boolean mask indexing), `torch.cuda.synchronize()`, and copies from non-pinned memory.
    3. Record CUDA events on the stream before and after, or `torch.cuda.synchronize()` on both sides and time with the CPU; warm up first and average over several runs.
    4. The caching allocator manages memory per stream: once a tensor is freed on the stream that allocated it, that memory may immediately be reused by a new allocation on that stream while another stream is still reading it. `record_stream` tells the allocator to wait until the other stream is done.
    5. `memory_allocated`: the bytes tensors currently occupy; `memory_reserved`: the total the caching allocator took from the driver (including cached free blocks), which is mostly what nvidia-smi shows. The difference is the cached free memory.

## Asynchronous execution and synchronization points {#异步执行与同步点}

A CUDA operator call does three things: check the arguments, allocate the output tensor (from the caching allocator, which is fast), and queue the kernel on the current stream. By the time it returns, the kernel has very likely not started. As long as the CPU does not **read** a result, it can keep submitting while the GPU runs behind, and the CPU's submission cost hides entirely behind the GPU's execution.

These operations need a value from the GPU and make the CPU **wait until the GPU has finished everything queued**:

| Operation | Why it synchronizes |
| --- | --- |
| `.item()`, `.tolist()`, `.cpu()`, `print(tensor)` | the value has to come back to the CPU |
| `torch.cuda.synchronize()` | an explicit wait |
| `nonzero()`, boolean mask indexing `x[mask]`, `torch.unique` | the output's shape depends on the data, so the result is needed before it can be allocated |
| `if tensor > 0:`, `while not done.all():` | Python's control flow needs an actual boolean |
| a copy to the GPU from pageable (non-pinned) memory | the driver copies into a temporary pinned buffer first |

An inference engine's scheduling loop needs to know "what token was sampled and which requests finished" every step, which is a synchronization point; overlap scheduling (mini-sglang's [overlap scheduling](minisgl://schedule/overlap/)) is essentially about deferring that synchronization until the next step's kernels have already been submitted.

**Timing** needs CUDA events (or a `synchronize` on both sides), or you only measure "the time to submit":

```python title="timing.py" run="no"
import torch

x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3):                     # warm-up: the first call pays for cuBLAS initialization and algorithm selection
    x @ w

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()                         # the event is queued too and timestamps when the GPU reaches it
for _ in range(10):
    y = x @ w
end.record()
end.synchronize()                      # wait for the end event
ms = start.elapsed_time(end) / 10
print(f"每次 {ms:.3f} ms，{2 * 8192**3 / ms / 1e9:.0f} TFLOPS")
```

## Streams: overlapping copies with compute {#stream让拷贝和计算重叠}

Move the chunk count and the stream count and watch the timeline overlap (the same tool as in the streams and events chapter):

<div class="aig-widget" data-widget="stream-overlap"></div>

Every operator goes on the **current stream** by default, and kernels within one stream run in order. To run two things in parallel (most typically "copy the next batch to the GPU" and "compute this one"), put them on different streams and establish the dependencies explicitly where they matter:

```python title="overlap_copy.py" run="no"
import torch

compute = torch.cuda.current_stream()
copy = torch.cuda.Stream()
w = torch.randn(4096, 4096, device="cuda")
batches = [torch.randn(4096, 4096).pin_memory() for _ in range(4)]   # pinned memory: the only way the copy is really asynchronous

next_gpu = batches[0].to("cuda", non_blocking=True)
for i in range(len(batches)):
    cur = next_gpu
    if i + 1 < len(batches):
        with torch.cuda.stream(copy):                   # prefetch the next batch on the copy stream
            next_gpu = batches[i + 1].to("cuda", non_blocking=True)
    y = cur @ w                                         # compute this batch on the compute stream
    compute.wait_stream(copy)                           # wait for next_gpu's copy before the next round uses it
    next_gpu.record_stream(compute)                     # tell the allocator this memory is still in use by the compute stream
torch.cuda.synchronize()
```

Two things that are easy to get wrong:

- **dependencies must be explicit**: `wait_stream`, or `event.record()` plus `stream.wait_event(event)`. Miss one and the compute may read data that is not finished copying, and it will only go wrong occasionally;
- **`record_stream`**: the caching allocator manages memory per stream. A tensor allocated on the copy stream, once freed, is assumed by the allocator to be reusable "as soon as the copy stream's work is done", without knowing the compute stream is still reading it. `record_stream(compute)` makes the allocator wait for the compute stream's current work before reusing that memory.

## The caching allocator {#缓存分配器}

`cudaMalloc` / `cudaFree` are slow and synchronizing, so PyTorch uses a **caching allocator**: freed memory is not returned to the driver but kept in pools by size for reuse (the C++ handbook's [allocators and memory pools](cpp://memory/allocators/) implements a simplified one). That gives two numbers:

- `torch.cuda.memory_allocated()`: the bytes tensors currently occupy;
- `torch.cuda.memory_reserved()`: the total bytes the allocator took from the driver (mostly what `nvidia-smi` shows).

The difference is the cached free blocks. "Plenty reserved yet still OOM" is usually fragmentation: the free blocks cannot be put together into one large contiguous allocation. The usual diagnostics and remedies:

```python title="memory_debug.py" run="no"
import torch

torch.cuda.memory._record_memory_history(max_entries=100_000)   # record the call stack of every allocation and free
torch.cuda.reset_peak_memory_stats()

x = torch.randn(4096, 4096, device="cuda")
y = x @ x
del x

print(f"allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB，"
      f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB，"
      f"峰值 {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")        # drop it into https://pytorch.org/memory_viz to see the timeline
```

- the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` has the allocator extend segments through virtual memory mappings, which cuts fragmentation sharply;
- inference engines allocate the whole KV cache at startup by `gpu_memory_utilization` and barely ask the allocator for large blocks at run time.

## CUDA Graphs {#cuda-graph}

A decode step has hundreds of kernels each computing for a few to a few dozen microseconds, so the CPU's submission cost (a few microseconds each) takes a considerable share. **CUDA Graphs** record a whole chain of kernels so it can be submitted once and replayed as a whole:

```python title="cuda_graph_decode.py" run="no"
import torch

model = torch.nn.Sequential(torch.nn.Linear(4096, 11008), torch.nn.SiLU(), torch.nn.Linear(11008, 4096)).cuda().half()
static_in = torch.zeros(8, 4096, device="cuda", dtype=torch.half)   # both capture and replay use this fixed memory

side = torch.cuda.Stream()                    # warm up on a non-default stream so the allocator and cuBLAS finish initializing
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side), torch.inference_mode():
    for _ in range(3):
        model(static_in)
torch.cuda.current_stream().wait_stream(side)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.inference_mode():
    static_out = model(static_in)             # captured, not executed

for step in range(5):
    new_input = torch.randn(8, 4096, device="cuda", dtype=torch.half)
    static_in.copy_(new_input)                # copy the new input into the fixed address
    g.replay()                                # the whole graph submitted at once
    token_logits = static_out.clone()         # the result is at the fixed output address
```

What is recorded is **a fixed sequence of kernels at fixed memory addresses**, so:

- the inputs have to be copied into the memory used at capture and the outputs read from fixed addresses; miss one input copy and the replay uses stale data;
- the shapes must be fixed, so inference engines capture one graph per common batch size and round the real batch up to the nearest;
- nothing synchronizing and no data-dependent control flow may be captured, and memory used inside the graph must not be freed outside it.

`torch.compile(mode="reduce-overhead")` does all this automatically; vLLM and SGLang manage several graphs themselves outside the model (see the Inference Systems handbook's [CUDA Graphs and torch.compile](serving://engine/graphs-compile/)).

!!! interview "Answering in an interview"
    On PyTorch's CUDA runtime: a GPU operator queues and returns, while `.item()`, `.cpu()`, data-dependent shapes and control flow all synchronize the CPU, so the inference hot path avoids them; time with CUDA events or a synchronization on both sides, after warming up; using a tensor across streams needs an explicit dependency and a `record_stream`, and asynchronous copies need pinned memory. Memory has two numbers: allocated is what tensors use and reserved is what the caching allocator took from the driver, and OOM with plenty reserved is usually fragmentation, for which there is the memory snapshot and `expandable_segments`. A CUDA Graph is captured at fixed addresses and shapes, with the inputs copied into fixed buffers.

## Exercises {#练习}

1. The sampling code below makes the CPU wait for the GPU once per step. Find the synchronization point and say how to rewrite it without one (hint: the finished check can stay on the GPU).

    ```python
    next_tokens = torch.argmax(logits, dim=-1)
    for i, tok in enumerate(next_tokens.tolist()):
        if tok == eos_id:
            finished[i] = True
    ```

??? success "Answer"
    `.tolist()` copies the result back to the CPU, which is a synchronization, and the Python loop after it can only start once the copy is done. The rewrite: compute the finished flags on the GPU as `done = next_tokens == eos_id`, copy them along with the tokens into a pinned buffer with `non_blocking=True`, and record an event;
    the scheduler then submits the next step's kernels first and calls `event.synchronize()` only when it actually needs the result. The copy and the synchronization then hide behind the next step's GPU time, which is what overlap scheduling does.

2. Why must a CUDA Graph replay's output be `static_out.clone()` rather than putting `static_out` itself into the result list?

??? success "Answer"
    `static_out` points at the fixed memory from capture time and the next `replay()` overwrites it. Keeping the reference leaves every element of the list pointing at the same memory holding the same (last) result. Copy what you want to keep; inference engines usually copy the part they need (the sampled token) elsewhere within the same step.

## Summary {#小结}

- [x] A GPU operator queues and returns; reading a result (`.item()`, `.cpu()`, data-dependent shapes and control flow) synchronizes, which the inference hot path avoids.
- [x] Time with CUDA events or a synchronization on both sides, after warming up.
- [x] Different streams run in parallel and dependencies must be explicit; using a tensor across streams needs `record_stream`; asynchronous copies need pinned memory.
- [x] The caching allocator: allocated is what is in use and reserved is what was taken; when fragmentation causes OOM, read the memory snapshot and turn on `expandable_segments`.
- [x] A CUDA Graph is captured at fixed addresses and shapes and submits the whole chain at once; copy the inputs into the fixed addresses and copy the outputs away promptly.
