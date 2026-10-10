# Stage 4 milestone: the same code on more GPUs, with faster kernels

<p class="lead">The engine built in the first three stages is complete, but it only uses one GPU and only PyTorch's operators. The six chapters of stage 4 let the same code split across several GPUs (tensor parallelism), switch to FlashInfer / FlashAttention on a GPU, record decode as a CUDA Graph and replay it, use custom kernels, run an MoE model, and finally measure what every optimisation is worth in one benchmark table. This page verifies "correct" on a CPU first: split into two ranks, or recorded and replayed as a graph, not one output token changes. The "fast" numbers need a real card.</p>

**What you hold at the end of this stage**

- The same `Engine`: with `tp_size=2` each rank holds half the weights and half the KV heads, the forward pass merges with all-reduce, and the output matches a single GPU;
- On a GPU, attention switches to FlashInfer / FlashAttention by itself, decode goes through CUDA Graph by itself, and KV writes and indexing use custom kernels;
- An MoE model (Qwen3-MoE) runs too, with a fused-MoE Triton kernel replacing the per-expert loop on a GPU;
- Chapter 21's benchmark table: each optimisation toggled on its own, with its throughput and latency.

## Run it first {#先跑起来}

```bash
cd minisgl && python examples/stage4_milestone.py
```

@@code examples/stage4_milestone.py@@

@@output stage4_milestone@@

## Reading these lines {#读这几行}

**With TP=2, `qkv_proj` goes from (4096, 1024) to (2048, 1024), and the KV pool's heads per layer from 8 to 4.** Each rank loads and stores only its half — which is exactly the problem tensor parallelism solves: when the model does not fit on one card, split every linear layer by rows or columns, let two cards compute half each, and merge with one all-reduce in between. The 4 tokens are identical to TP=1: with the right split, it is mathematically the same matrix multiply. On a CPU the two processes go through gloo and are no faster; on real cards they go through NCCL, the weight-reading bandwidth doubles, and decode speeds up.

**CUDA Graph emulation: 3 requests padded to 4, replayed 5 times, same output.** CUDA Graph requires every replay to use fixed tensor shapes and addresses, so the batch is padded to a recorded size and every input is first copied into fixed buffers. The CPU emulation does one thing: enforce that "fixed buffers" discipline — skip copying any one input and the output is wrong at once (chapter 18 demonstrates this on purpose). On a real card that discipline buys the removal of dozens of kernel-launch overheads from every decode step.

**On a CPU, this is as far as it goes.** The other three chapters of this stage — FlashInfer / FlashAttention, custom CUDA kernels, fused MoE — only have same-interface stand-ins on a CPU: fakes, the CPU simulator, the Triton interpreter. They verify logic and interfaces, not speed. With a card, `gpu_check.py minisgl` in the repository root runs these chapters on real hardware; chapter 21's benchmark table is the real milestone of this stage.

## What each chapter adds {#每一章加了什么}

| Optimisation | What it solves | How it is verified on a CPU | Where |
| --- | --- | --- | --- |
| tensor parallelism | the model does not fit on one card; several cards read the weights together | two processes over gloo, output identical to one card | [Tensor parallelism](tensor-parallel.md) |
| FlashInfer / FlashAttention | never materialise the attention score matrix; read paged KV directly | same-interface PyTorch fakes | [GPU attention backends](gpu-attention.md) |
| CUDA Graph | the CPU overhead of dozens of kernel launches per decode step | `EmulatedGraph`, enforcing fixed buffers | [CUDA Graph](cuda-graph.md) |
| custom kernels | launch overhead and memory traffic of small operations such as KV writes and vocabulary indexing | the CUDA handbook's CPU simulator | [Custom CUDA kernels](kernels.md) |
| fused MoE | the per-expert loop becomes one grouped matrix multiply | Triton interpreter mode | [MoE and fused MoE](moe.md) |
| benchmarks | what each optimisation is actually worth | real hardware only | [Benchmarks and ablations](benchmark.md) |

## How to read these six chapters {#怎么读这六章}

[Tensor parallelism](tensor-parallel.md) and [CUDA Graph](cuda-graph.md) can be verified completely on a CPU; read them first. Read [GPU attention backends](gpu-attention.md) and [custom kernels](kernels.md) alongside the matching kernel chapters of the [CUDA handbook](cuda://); [MoE](moe.md) can wait until last. After the six chapters, if you have a card, take the engine to 60% of the official build by the bar in [the project](../wrap/assignment.md) — that is where this book really ends.

!!! abstract "Checkpoint: after this stage"
    - [ ] Say which of the four linear-layer splits applies to which layer, and why `o_proj` and `down_proj` are followed by an all-reduce;
    - [ ] Explain why CUDA Graph pads the batch, and why skipping one input copy produces a wrong answer rather than an error;
    - [ ] Explain what FlashInfer's plan and run steps each do;
    - [ ] Run `gpu_check.py minisgl` on a real card, and explain where every difference in chapter 21's benchmark table comes from.

## Summary {#小结}

- [x] Tensor parallelism: half the weights and half the KV heads per rank, merged with all-reduce, output identical to one card.
- [x] CUDA Graph: fixed shapes and fixed buffers; the CPU emulation verifies only that discipline.
- [x] Attention backends, custom kernels and fused MoE can only be verified for interface and logic on a CPU; speed needs a real card.
- [x] Chapter 21's benchmark table is the real milestone of this stage.
