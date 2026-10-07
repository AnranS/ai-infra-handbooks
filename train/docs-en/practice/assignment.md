# The assignment: DDP, ZeRO-1 and activation recomputation

<p class="lead">Every chapter in this book gives a minimal implementation of one kind of parallelism. The assignment asks you to close the book, work from the interfaces alone, and write bucketed DDP, ZeRO-1 and activation recomputation yourself, making them match a single process elementwise when combined in training; then train a model with them and compare the measured memory and throughput against the first chapter's budget.</p>

The code and the instructions are in the repository at [`assignments/a2-trainsys/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/assignments/a2-trainsys).

## What to do {#要做什么}

| File | Contents | Chapter | Check (several processes on the CPU, gloo) |
| --- | --- | --- | --- |
| `trainsys/ddp.py` | `BucketedDDP`: broadcast the parameters at construction, bucket them in backward order, all-reduce asynchronously as soon as a bucket is full | [DDP](../data/ddp.md) | matches a single process; an asynchronous all-reduce really is fired during the backward pass |
| `trainsys/zero1.py` | `ZeRO1`: flatten the parameters, split the optimizer states evenly, all-gather after the update | [ZeRO and FSDP](../data/zero-fsdp.md) | matches single-process AdamW; each rank's state is about 1 / world |
| `trainsys/recompute.py` | `checkpoint(fn, *args)`: a custom autograd function that recomputes in the backward pass | [Mixed precision](mixed-precision.md#省激活的其他手段) | the gradients match; the bytes saved for the backward pass are at least halved |
| Combined | all three training a network with residual blocks | — | matches a single process |

The check script does not care about your internals: DDP's overlap is judged by intercepting `dist.all_reduce` and seeing whether it is called with `async_op=True` before `backward()` returns, and recomputation's effect is judged by counting the bytes the forward pass saved with `torch.autograd.graph.saved_tensors_hooks`.

```bash
cd assignments/a2-trainsys
python run_checks.py                 # all 4 checks pass
```

## The report {#报告}

Train the model from [the large-model handbook's assignment](llm://training/assignment/) with these three components across 2 to 4 processes and write a one-page report:

1. **The memory budget**: how much the parameters, gradients, optimizer states and activations each take, against [the overview](../basics/overview.md)'s formulas.
2. **The bucket size**: how the time per step changes from 16 KB to 16 MB, and why (see the timeline model in [DDP](../data/ddp.md)).
3. **Recomputation's cost**: the extra time and the memory saved, and whether they match the estimate of one extra forward pass.

## Going further {#延伸}

- Turn it into ZeRO-2: partition the gradients too, replace the all-reduce with a reduce-scatter, and verify the volume is unchanged.
- Allocate the buckets directly in a contiguous buffer, saving the concatenation's copy.
- Move to a GPU with NCCL and confirm with a profiler that the communication really does overlap the backward pass.
