# PyTorch in a Hurry

<p class="lead">Eight chapters to get fluent with PyTorch: tensors and shapes, indexing and masks, autograd, nn.Module, data and the training loop, saving and reproducing, debugging and speed. This book is only about <strong>using</strong> it — every section is a snippet that finishes in seconds plus its real output, and by the end you can read and write the training code in mainstream projects. For how these things are <strong>implemented</strong>, see the CUDA handbook's "Frameworks and compilers" chapters; to put them to work on a full training run, see <em>Train a Small Model</em>.</p>

## Who this book is for {#这本书适合谁}

- You know Python and understand what a neural network computes, but your PyTorch stops at "I can adapt someone else's training script".
- Reading source code, you keep getting stuck on shapes, `gather`, `detach`, `state_dict` and the like.
- You want to diagnose "the shapes do not match", "the gradient is None" and "memory keeps growing" by yourself.

By the end, working through it, you should be able to:

- See at a glance how the shapes move through a piece of tensor code, and tell a view from a copy.
- Write the operations inside attention and loss functions with `gather`, `scatter_`, `masked_fill` and `einsum`.
- Explain what `no_grad`, `detach` and `eval()` each control, and why none of them substitutes for another.
- Write a training loop with evaluation, scheduling and clipping, and resume it after an interruption with identical results.
- Read error messages, and find the slowest operator with the profiler.

## The eight chapters {#八章的路线}

<div class="roadmap" markdown>

| Chapter | What you do | What you can do afterwards | Suggested time |
| --- | --- | --- | --- |
| [1. Tensors](tensor.md) | shape, dtype, device, and which operations share memory | stop getting stuck on dtype and device errors | 1 hour |
| [2. Working with shapes](shape.md) | view / reshape / permute, broadcasting, einsum | know how shapes move, and write multi-head attention's subscripts | 2 hours |
| [3. Indexing, reductions and masks](indexing.md) | views versus copies, gather / scatter_, causal masks, dim and keepdim | write the lookups inside cross-entropy and attention | 2 hours |
| [4. Using autograd](autograd.md) | backward, gradient accumulation, no_grad and detach | explain why a gradient is None, and know which in-place ops break | 2 hours |
| [5. nn.Module](module.md) | parameters and buffers, state_dict, train / eval | write models that are clear, saveable and loadable | 2 hours |
| [6. Data and the training loop](training.md) | Dataset, DataLoader, collate_fn, and a complete training run | write a working training script on your own | 3 hours |
| [7. Saving, loading and reproducing](checkpoint.md) | what belongs in a checkpoint, seeds and determinism | resume after an interruption with identical results | 1 hour |
| [8. Debugging and speed](debug.md) | reading errors, the classic traps, inference_mode, autocast, the profiler | diagnose shape, gradient and performance problems yourself | 2 hours |

</div>

The eight chapters are independent; read whichever one you are stuck on. All the code runs in seconds on a CPU.

## Where to go next {#接下来往哪走}

| If you want to | Go to |
| --- | --- |
| Put all of this to work and train a small language model end to end | [Train a Small Model](scratch://) |
| What a tensor looks like in memory, how autograd builds the graph, what torch.compile does | [the CUDA handbook's "Frameworks and compilers"](cuda://framework/tensor/) |
| Why each part of a Transformer is designed the way it is | [LLM internals](llm://) |
| What to do when one GPU is not enough: DDP, ZeRO, tensor parallelism | [distributed training](train://) |
| Writing more idiomatic Python | [Advanced Python](python://) |

The chapter-by-chapter route through every handbook is in the [learning roadmap](root://roadmap/), and the job-hunting schedule is in the [17-week plan](root://plan/).

## How it is verified {#怎么验证的}

- Every snippet on every page really runs on **CPU PyTorch**, and the output on the page matches the run line by line.
- The error messages are real too — every `RuntimeError` in this book can be reproduced.

```bash
# needs CPU PyTorch (2.4 or newer) and numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```
