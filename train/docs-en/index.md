# The distributed training handbook

<p class="lead">An inference role still has to understand the parallelism on the training side: interviews ask about ZeRO and Megatron, reinforcement-learning training ties an inference engine to a training framework, and the tensor, expert and context parallelism used in inference all came from training. This handbook starts from what will not fit on one card, implements data parallelism, ZeRO, tensor parallelism, pipeline parallelism, context parallelism and expert parallelism from scratch one by one, and lines each of them up against a single process on the CPU using multiple processes.</p>

## Who this handbook is for {#这份手册适合谁}

- You have never trained a language model by hand: start with [training a small model from scratch](scratch/data.md) and walk the whole pipeline through on your own computer.
- You know the Transformer's structure and can write a training loop in PyTorch, but have never done multi-GPU training.
- Or you have used DeepSpeed, Megatron-LM or FSDP, but cannot say what each step communicates or where the memory is saved.
- You work on inference systems and need to understand the training side's parallelism (reinforcement-learning training, weight synchronisation, the tensor-parallel and expert-parallel implementations shared with inference).

After reading and working through this handbook you should be able to:

- Compute the memory budget per card (parameters, gradients, optimizer states, activations) and the communication volume per step for any model and cluster configuration.
- Say clearly what data parallelism, ZeRO-1/2/3, tensor parallelism, sequence parallelism, pipeline parallelism, context parallelism and expert parallelism each partition, what they communicate and what they cost.
- Write DDP from scratch (bucketing with overlapped communication), ZeRO-1, a tensor-parallel MLP including the backward pass, Ulysses attention and an expert-parallel mixture-of-experts layer, and verify each against the single-card result.
- Choose a sensible combination of parallelism for a specific model and cluster, and explain why.

## The route through {#学习路线}

<div class="roadmap" markdown>

| Stage | Chapters | What you can do afterwards | Suggested time |
| --- | --- | --- | --- |
| 0. From scratch | [The corpus and the tokenizer](scratch/data.md) · [The model and the training loop](scratch/model.md) · [Training faster and larger](scratch/scale.md) | train a small model on your own computer that writes in the style of a classical novel, and know which sums to do when scaling up | 2 to 3 days |
| 1. Basics | [Overview: the memory budget and the time model](basics/overview.md) · [Collective primitives](basics/collectives.md) | work out the memory and the communication, and know where the bottleneck is | 2 to 3 days |
| 2. Data parallelism | [DDP](data/ddp.md) · [ZeRO and FSDP](data/zero-fsdp.md) | write DDP with bucketed overlap, and ZeRO-1 | 2 to 3 days |
| 3. Model parallelism | [Tensor and sequence parallelism](model/tensor-sequence.md) · [Pipeline parallelism](model/pipeline.md) · [Context parallelism](model/context.md) · [Mixture of experts and expert parallelism](model/moe-ep.md) | explain each kind's partitioning and communication, and write a minimal implementation of it | 1 week |
| 4. Precision and strategy | [Mixed precision and FP8](practice/mixed-precision.md) · [Combining 3D and 5D parallelism](practice/strategy.md) · [Training frameworks and reinforcement-learning systems](practice/frameworks-rl.md) | pick a configuration for a specific case, and read Megatron, DeepSpeed and verl | 3 to 4 days |
| 5. Optimizers, stability and RL algorithms | [AdamW, Muon and distributed optimizers](algo/optimizer.md) · [Training stability](algo/stability.md) · [Reinforcement-learning algorithms in depth](algo/rl-algorithms.md) | explain the newer optimizers, the root causes of a loss spike and what to do about them, and the reinforcement-learning algorithms after GRPO | 3 to 4 days |

</div>

The chapter-by-chapter route through every handbook is in [the roadmap](root://roadmap/), and the week-by-week schedule is in [the plan](root://plan/); this book is week 9 of the plan, studied alongside [distributed inference](serving://distributed/tensor-parallel/) in the inference-systems handbook.

## How it was verified {#怎么验证的}

- Every script with a file name on it actually runs on **the CPU build of PyTorch**, and the output on the page matches the run line by line.
- The multi-process scripts are launched with `torchrun --standalone --nproc-per-node N` and the communication backend is **gloo** (CPU); moving to a GPU only takes changing the backend to `nccl` and putting the tensors on `cuda`.
- Every kind of parallelism is lined up item by item against the single-process computation: the forward output, the loss, and **the gradients**.

```bash
# the CPU build of PyTorch is required (2.4 or newer)
pip install torch --index-url https://download.pytorch.org/whl/cpu
torchrun --standalone --nproc-per-node 4 collectives.py
```
