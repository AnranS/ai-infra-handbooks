# From scratch (3): training faster and larger

<p class="lead">The small model is trained. This chapter answers what comes next: where a training step's time goes and how much of the hardware's throughput is used; what to do when a larger batch will not fit in memory; how to train across several cards and still match one card's result; how much better a larger model gets; and, on real GPUs, how many cards and hours a 1B model takes. Each question is measured on the same model on a CPU first, then extrapolated to GPUs and large models, with a pointer to the chapter in this handbook that covers it.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Roughly how many floating-point operations does training on one token take? How is the model FLOPs utilization computed?
    2. Why is gradient accumulation equivalent to a large batch? How is the loss scaled?
    3. Under DDP, how does each rank's data differ? Who saves the checkpoint?
    4. With the same amount of data, is a larger model always better? What does Chinchilla's 20 tokens per parameter mean?
    5. Training a 1.3B model on the compute-optimal amount of data, how long does it take on 8 H100s?

??? success "Answers for the self-test (answer first, then open this)"
    1. About $6N$ ($2N$ forward and $4N$ backward). The model FLOPs utilization is tokens per second times $6N$ divided by the hardware's peak; training a Transformer on a GPU is generally 30% to 50%.
    2. The gradient is the average of the per-sample gradients, so splitting a large batch into several parts, backpropagating each and accumulating the gradients matches computing the whole batch's gradient at once. Each part's loss has to be divided by the number of parts for the accumulation to be the average gradient.
    3. Each rank reads different data (offset by rank, or through a DistributedSampler); the gradients are all-reduced to an average so the parameters stay identical, which is why only rank 0 has to write the logs and save the checkpoint.
    4. Not necessarily: with the data fixed, past a certain size the returns diminish and the overfitting worsens. What Chinchilla says is that for a given compute budget, the lowest loss comes from training on about 20 tokens per parameter, with the parameter count and the data growing in proportion.
    5. At 20 tokens per parameter that is 26B tokens, about $6 \times 1.3 \times 10^9 \times 2.6 \times 10^{10} \approx 2 \times 10^{20}$ operations; on 8 H100s at a reasonable utilization, about 18 hours (this chapter estimates 17.8).

## Measure first: where the time goes {#先测量时间花在哪里}

Training on one token takes about **6 times the parameter count** in operations: the forward pass is one multiply-add per parameter (2 operations), and the backward pass computes the input's gradient and the weights' gradient once each, 3 times in all. Attention's $QK^\top$ and $PV$ are not in the parameters and have to be added separately (a small share with a short context). Use it to convert tokens per second into throughput actually achieved:

```python title="speed.py"
import time

import torch

from model import GPT, GPTConfig

cfg = GPTConfig()
data = torch.load("tokens.pt")["train"].long()
B, T = 16, cfg.seq_len
i = torch.randint(0, len(data) - T - 1, (B,), generator=torch.Generator().manual_seed(0))
x = torch.stack([data[j:j + T] for j in i])
y = torch.stack([data[j + 1:j + T + 1] for j in i])


def flops_per_token(model):
    """前向 + 反向 ≈ 6 × 参数量，再加上注意力里 QKᵀ 和 PV 的 12·L·T·d（前向 2 次矩阵乘 × 2 FLOP，反向再 2 倍）"""
    return 6 * model.num_params() + 12 * cfg.n_layer * T * cfg.d_model


def measure(name, model, autocast=False, steps=10):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for s in range(steps + 3):
        if s == 3:
            t0 = time.perf_counter()                          # the first few steps are a warm-up (where torch.compile does its compiling)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dt = (time.perf_counter() - t0) / steps
    tps = B * T / dt
    print(f"{name:22s} 每步 {dt * 1e3:6.1f} ms，{tps:8.0f} token/s，实际算力 {tps * flops_per_token(model) / 1e9:5.1f} GFLOPS")


torch.manual_seed(0)
measure("fp32", GPT(cfg))
torch.manual_seed(0)
measure("bf16 autocast", GPT(cfg), autocast=True)
torch.manual_seed(0)
t0 = time.perf_counter()
compiled = torch.compile(GPT(cfg))
measure("fp32 + torch.compile", compiled)
```

Here is the result on 2 CPU cores of a server (the times differ on every run, so they are not checked line by line):

```text
fp32                   每步  224.2 ms，    9136 token/s，实际算力 106.0 GFLOPS
bf16 autocast          每步  619.4 ms，    3306 token/s，实际算力  38.4 GFLOPS
fp32 + torch.compile   每步  184.4 ms，   11108 token/s，实际算力 128.9 GFLOPS
```

- **bf16 is nearly 3 times slower on this CPU**: it has no bf16 matrix instructions, so autocast only adds type conversions. Low precision is only fast when the hardware supports it; on a GPU's Tensor Cores bf16 has more than ten times fp32's throughput, which is what makes mixed precision the default for training (see [Mixed precision](../practice/mixed-precision.md)).
- **torch.compile is about 20% faster**: it fuses the scattered elementwise operations of RMSNorm, SwiGLU and RoPE into a few kernels and reads and writes memory fewer times (the principle is in [torch.compile](cuda://framework/compile/) in the CUDA handbook). The gain is usually larger for a small model on a GPU, where the kernel launch overhead is more visible.
- **The model FLOPs utilization** is the achieved throughput over the hardware's peak. Training a Transformer on a GPU is generally 30% to 50%, and below 20% means something is clearly wasted: too small a batch, data loading that cannot keep up, communication not overlapped with computation, or too many small kernels.

## Not enough memory: gradient accumulation {#显存不够梯度累积}

![Figure: gradient accumulation, where several micro-batches' gradients are accumulated before one update](../assets/figures/grad-accum.svg){.aig-svg}

A larger batch means less noise in the gradient but more memory for the activations. **Gradient accumulation** splits a large batch into parts computed one after another, accumulating the gradients before one update, which is mathematically equivalent, provided each part's loss is divided by the number of parts:

```python title="grad_accum.py"
import torch

from model import GPT, GPTConfig

data = torch.load("tokens.pt")["train"].long()
cfg = GPTConfig(d_model=64, n_head=2, n_layer=2, seq_len=64)
i = torch.randint(0, len(data) - 65, (32,), generator=torch.Generator().manual_seed(0))
x = torch.stack([data[j:j + 64] for j in i])
y = torch.stack([data[j + 1:j + 65] for j in i])


def grads(micro_batches):
    torch.manual_seed(0)
    model = GPT(cfg)
    for xs, ys in zip(x.chunk(micro_batches), y.chunk(micro_batches)):
        _, loss = model(xs, ys)
        (loss / micro_batches).backward()                     # each micro-batch's loss divided by the number of parts: the accumulated gradients are then the large batch's average
    return torch.cat([p.grad.flatten() for p in model.parameters()])


full = grads(1)
for k in (2, 4, 8):
    diff = (grads(k) - full).abs().max().item()
    print(f"32 条分成 {k} 份累加：与一次算 32 条的梯度最大差小于 1e-6：{diff < 1e-6}（梯度最大值 {full.abs().max():.2f}）")
```

```text title="output"
32 条分成 2 份累加：与一次算 32 条的梯度最大差小于 1e-6：True（梯度最大值 0.06）
32 条分成 4 份累加：与一次算 32 条的梯度最大差小于 1e-6：True（梯度最大值 0.06）
32 条分成 8 份累加：与一次算 32 条的梯度最大差小于 1e-6：True（梯度最大值 0.06）
```

The price is time: split into $k$ parts, you run $k$ forward and backward passes in series. This and activation recomputation are the two basic ways to save memory (the memory budget is in [the overview](../basics/overview.md)). Note that an operation across samples such as BatchNorm breaks the equivalence; the LayerNorm and RMSNorm a Transformer uses do not have this problem.

## Several cards: DDP {#多张卡ddp}

Data parallelism (DDP): each card holds a complete model and handles different data, and the gradients are averaged in the backward pass (an all-reduce), so every card's parameters stay the same. Compared against a single process whose batch each step is all of the ranks' data concatenated:

```python title="ddp_train.py" torchrun="2"
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

dist.init_process_group("gloo")                               # on a GPU this becomes "nccl", with the model and the data on cuda:{local_rank}
rank, world = dist.get_rank(), dist.get_world_size()
data = torch.load("tokens.pt")["train"].long()
cfg = GPTConfig(d_model=64, n_head=2, n_layer=2, seq_len=64)
PER_RANK, STEPS = 8, 30


def batch(step, r):
    """第 step 步、第 r 个 rank 的数据：每个 rank 拿不同的片段（真实训练里是数据集的不同分片）"""
    g = torch.Generator().manual_seed(1000 * step + r)
    i = torch.randint(0, len(data) - 65, (PER_RANK,), generator=g)
    return torch.stack([data[j:j + 64] for j in i]), torch.stack([data[j + 1:j + 65] for j in i])


torch.manual_seed(0)                                           # the same initialisation on every rank (DDP also broadcasts from rank 0 at construction)
model = DDP(GPT(cfg))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
for step in range(STEPS):
    x, y = batch(step, rank)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()                                            # the backward pass all-reduces the gradients by bucket (taking the average)
    opt.step()

if rank == 0:                                                  # the control: a single process whose batch each step is all of the ranks' data concatenated
    torch.manual_seed(0)
    single = GPT(cfg)
    opt1 = torch.optim.AdamW(single.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    for step in range(STEPS):
        xs, ys = zip(*(batch(step, r) for r in range(world)))
        _, loss1 = single(torch.cat(xs), torch.cat(ys))
        opt1.zero_grad(set_to_none=True)
        loss1.backward()
        opt1.step()
    diff = max((a - b).abs().max().item() for a, b in zip(model.module.parameters(), single.parameters()))
    print(f"{world} 个进程，每个每步 {PER_RANK} 条：训练 {STEPS} 步后与单进程（每步 {PER_RANK * world} 条）的参数最大差小于 1e-4：{diff < 1e-4}")
    torch.save(model.module.state_dict(), "ddp_model.pt")   # checkpoint only on rank 0, saving .module without the DDP wrapper
dist.destroy_process_group()
```

```text title="output"
2 个进程，每个每步 8 条：训练 30 步后与单进程（每步 16 条）的参数最大差小于 1e-4：True
```

- **Each rank's data has to differ**, or several cards compute the same gradient and the work is wasted. A real data loader partitions the dataset by rank (`DistributedSampler`, or reading by shard).
- **The global batch is the per-card batch times the card count times the accumulation steps**. Adding cards has to keep the global batch the same (or adjust the learning rate to match), or the training dynamics change.
- **Logging, validation and checkpointing happen only on rank 0**, saving `model.module` without the DDP wrapper.
- The difference comes from the order of the floating-point accumulation (the all-reduce sums in a different order than a single process). How DDP buckets the gradients and overlaps them with the backward pass is in [Data parallelism and DDP](../data/ddp.md); when the model grows beyond one card's complete parameters and optimizer states, [ZeRO and FSDP](../data/zero-fsdp.md) partition them.

## Making the model larger: how much better does it get {#做大模型效果能好多少}

The same data and the same 400 steps, training four models of different sizes:

```python title="scaling.py"
import math

import torch

from model import GPT, GPTConfig

torch.set_num_threads(8)                                       # this experiment trains 4 models, so use a few more threads
data = torch.load("tokens.pt")
train_data, val_data = data["train"].long(), data["val"].long()
T, B, STEPS = 128, 16, 400
val_gen = torch.Generator().manual_seed(123)
val_batches = []
for _ in range(8):
    i = torch.randint(0, len(val_data) - T - 1, (32,), generator=val_gen)
    val_batches.append((torch.stack([val_data[j:j + T] for j in i]), torch.stack([val_data[j + 1:j + T + 1] for j in i])))


def run(d_model, n_layer):
    torch.manual_seed(0)
    model = GPT(GPTConfig(d_model=d_model, n_layer=n_layer, n_head=d_model // 32, seq_len=T))
    decay = [p for p in model.parameters() if p.dim() >= 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}],
                            lr=3e-3, betas=(0.9, 0.95))
    gen, curve = torch.Generator().manual_seed(0), []
    for step in range(STEPS):
        lr = 3e-3 * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr
        i = torch.randint(0, len(train_data) - T - 1, (B,), generator=gen)
        x = torch.stack([train_data[j:j + T] for j in i])
        y = torch.stack([train_data[j + 1:j + T + 1] for j in i])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                curve.append(sum(model(vx, vy)[1].item() for vx, vy in val_batches) / len(val_batches))
            model.train()
    return model.num_params(), curve, loss.item()


print("  d  层数  参数量    验证 loss：第 100    200    300    400 步   最后的训练 loss")
for d, L in [(64, 2), (96, 3), (128, 4), (192, 4)]:
    n, curve, train_loss = run(d, L)
    print(f"{d:3d}  {L:3d}  {n / 1e6:5.2f}M        " + "  ".join(f"{v:5.2f}" for v in curve) + f"      {train_loss:5.2f}")
```

```text title="output"
  d  层数  参数量    验证 loss：第 100    200    300    400 步   最后的训练 loss
 64    2   0.62M         6.85   6.63   6.52   6.49       5.83
 96    3   1.12M         6.79   6.52   6.39   6.34       5.65
128    4   1.80M         6.74   6.47   6.33   6.26       5.52
192    4   3.34M         6.70   6.44   6.28   6.22       5.41
```

- **The larger the model, the lower the validation loss**, at every stage; but the returns diminish, with 5 times the parameters giving only 0.27 less loss.
- **The gap between the training and validation losses widens with the model's size**: a larger model memorises more. Our data is only 410,000 tokens, so all four models are already in the not-enough-data regime.
- This is exactly what **scaling laws** study: given a compute budget $C \approx 6ND$, how should the parameter count $N$ and the data $D$ be divided? Chinchilla's conclusion is that the two should grow in proportion, at about **20 tokens per parameter**. At that ratio our 1.8M model wants 36 million tokens, 90 times what we have, so enlarging the model further on this little data gains less and less. Real large models are often trained far beyond 20 tokens per parameter (a 7B model on trillions of tokens), because inference cost is proportional to the parameter count and spending more training compute for a smaller, stronger model pays off.

## On a GPU {#放到-gpu-上}

The GPU training script: bf16 automatic mixed precision, `torch.compile`, DDP and gradient accumulation. It needs an NVIDIA GPU, so only the syntax is checked here:

```python title="train_gpu.py" run="no"
"""GPU 版训练脚本：bf16 自动混合精度 + torch.compile + DDP + 梯度累积。
单卡：python train_gpu.py；8 卡：torchrun --standalone --nproc-per-node 8 train_gpu.py"""
import contextlib
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

distributed = "RANK" in os.environ
if distributed:
    dist.init_process_group("nccl")
    rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
else:
    rank, world, local_rank = 0, 1, 0
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
torch.backends.cuda.matmul.allow_tf32 = True                  # the remaining fp32 matrix multiplies use the Tensor Cores too (TF32)

cfg = GPTConfig(d_model=512, n_layer=8, n_head=8, seq_len=512)   # about 29M parameters
MICRO, ACCUM, STEPS, LR, WARMUP = 32, 4, 2000, 1e-3, 100
global_batch = MICRO * ACCUM * world                         # the global batch per step (in sequences)
data = torch.load("tokens.pt")                                # a larger corpus is needed to really use a GPU (see the text)
train_data = data["train"].long()
gen = torch.Generator().manual_seed(1000 + rank)              # each rank takes different data


def get_batch():
    i = torch.randint(0, len(train_data) - cfg.seq_len - 1, (MICRO,), generator=gen)
    x = torch.stack([train_data[j:j + cfg.seq_len] for j in i])
    y = torch.stack([train_data[j + 1:j + cfg.seq_len + 1] for j in i])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


torch.manual_seed(0)
raw = GPT(cfg).to(device)
model = torch.compile(raw)
if distributed:
    model = DDP(model, device_ids=[local_rank])
decay = [p for p in raw.parameters() if p.dim() >= 2]
no_decay = [p for p in raw.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.95), fused=True)
flops_per_token = 6 * raw.num_params() + 12 * cfg.n_layer * cfg.seq_len * cfg.d_model
peak = 989e12                                                 # the H100's dense bf16 throughput; the A100's is 312e12

t0 = time.perf_counter()
for step in range(STEPS):
    lr = LR * min(1.0, (step + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
    for group in opt.param_groups:
        group["lr"] = lr
    for micro in range(ACCUM):
        x, y = get_batch()
        sync = (not distributed) or micro == ACCUM - 1       # synchronise the gradients only on the last micro-batch
        ctx = contextlib.nullcontext() if sync else model.no_sync()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
            (loss / ACCUM).backward()
    torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if rank == 0 and (step + 1) % 50 == 0:
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 50
        tps = global_batch * cfg.seq_len / dt
        print(f"step {step + 1}  loss {loss.item():.3f}  {tps:,.0f} token/s  "
              f"MFU {tps * flops_per_token / (world * peak):.0%}  显存 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        t0 = time.perf_counter()

if rank == 0:
    torch.save({"model": raw.state_dict(), "optimizer": opt.state_dict(), "step": STEPS}, "ckpt_gpu.pt")
if distributed:
    dist.destroy_process_group()
```

A few differences from the CPU version:

- `torch.autocast("cuda", dtype=torch.bfloat16)` puts the matrix multiplies in bf16 while the weights and optimizer states stay fp32 (the master weights); `allow_tf32` puts the remaining fp32 matrix multiplies on the Tensor Cores too.
- Under gradient accumulation, the earlier micro-batches use `model.no_sync()` to skip the gradient synchronisation and only the last one all-reduces, which cuts the communication to $1/k$.
- The data is copied to the GPU asynchronously with `pin_memory` and `non_blocking`, overlapping the computation.
- The throughput, the utilization and the peak memory are printed every 50 steps; those three numbers are where tuning starts.
- This book's corpus is only 410,000 tokens, and this 29M model on 8 cards reads it several times over in one step. To really use a GPU you need a far larger corpus (a subset of an open pretraining corpus, say).

Finally, estimating the resources to make the model larger at the compute-optimal data (20 tokens per parameter):

```python title="budget.py"
# scaling the tutorial's model up: estimating the compute, the time and the memory (H100: 989 dense bf16 TFLOPS, 80 GB)
PEAK, MFU = 989e12, 0.4


def estimate(name, n_params, gpus, d_model, n_layer, seq_len, micro_batch):
    tokens = 20 * n_params                                      # Chinchilla: about 20 tokens per parameter at compute-optimal
    flops = 6 * n_params * tokens                               # forward plus backward is about 6 x the parameter count x the token count
    hours = flops / (gpus * PEAK * MFU) / 3600
    states = 16 * n_params                                      # bf16 weights and gradients plus fp32 master weights plus Adam's two moments
    acts = 34 * seq_len * micro_batch * d_model * n_layer       # about 34 s b h bytes of activations per layer (bf16, FlashAttention, no recomputation)
    print(f"{name:14s} {tokens / 1e9:6.1f}B token  {flops:8.1e} FLOP  {gpus:3d} 卡 {hours:6.1f} 小时   "
          f"模型状态 DDP {states / 1e9:5.1f} GB / FSDP {states / gpus / 1e9:5.2f} GB，激活 {acts / 1e9:5.1f} GB")


print("模型              数据量        总计算量        时间（MFU 40%）       每卡显存")
estimate("124M（GPT-2）", 124e6, 1, 768, 12, 1024, 16)
estimate("1.3B", 1.3e9, 8, 2048, 24, 2048, 8)
estimate("7B", 7e9, 64, 4096, 32, 4096, 2)
```

```text title="output"
模型              数据量        总计算量        时间（MFU 40%）       每卡显存
124M（GPT-2）       2.5B token   1.8e+18 FLOP    1 卡    1.3 小时   模型状态 DDP   2.0 GB / FSDP  1.98 GB，激活   5.1 GB
1.3B             26.0B token   2.0e+20 FLOP    8 卡   17.8 小时   模型状态 DDP  20.8 GB / FSDP  2.60 GB，激活  27.4 GB
7B              140.0B token   5.9e+21 FLOP   64 卡   64.5 小时   模型状态 DDP 112.0 GB / FSDP  1.75 GB，激活  36.5 GB
```

- At GPT-2's 124M scale, one H100 finishes in an hour or two, which is a good starting point for reproducing it yourself.
- 1.3B takes under a day on 8 cards with DDP, with each card holding 20.8 GB of model states and 27 GB of activations; adding the output layer's logits and the various temporary buffers, the micro-batch cannot double again without turning on activation recomputation.
- 7B's model states (112 GB) do not fit on one card at all and have to be partitioned across cards with FSDP or ZeRO-3 (1.75 GB each), with the activations controlled by recomputation or a smaller micro-batch. From here on, you are in the [parallelism strategies](../practice/strategy.md) that the rest of this handbook covers.

!!! interview "How to answer in an interview"
    Asked how you would plan to train a model of about 1B on 8 H100s: **compute the budget** ($C \approx 6ND$, taking $D \approx 20B$ at 20 tokens per parameter; at 40% utilization, about ten hours), **compute the memory** (model states at 16 bytes per parameter, about 16 GB, plus the activations; DDP if it fits and FSDP or activation recomputation if not), **set the batch** (a global batch of a few million tokens, made up of the per-card batch times the card count times the accumulation steps), **small before large** (get the data, the tokenizer, the training loop, resumption and evaluation working on a small model and sweep the learning rate before scaling up), and **monitor** (the loss, the gradient norm, the throughput, the utilization, the memory, with generated samples and evaluations looked at regularly).

## Exercises {#练习}

**1. Sweep the learning rate.** Using `scaling.py`'s form, try 1e-3, 3e-3 and 1e-2 on the 1.8M model. Which is best? Does the best learning rate change on the 3.3M model?

??? success "The approach"
    You usually see a U shape: too small a learning rate and 400 steps do not go far, too large and the loss is more erratic or diverges. The best learning rate generally gets smaller as the model gets wider. In real training the sweep happens on a small model and transfers to the large one by an empirical rule (or a parameterisation like muP), because sweeping on a large model is too expensive.

**2. Why will the utilization not rise?** A GPU training job has a model FLOPs utilization of only 15%. List the things you would check.

??? success "Answer"
    - **The batch is too small**: the matrix multiplies are too small to fill the Tensor Cores. Increase the micro-batch or the sequence length.
    - **Data loading**: the GPU is waiting on the CPU to prepare data. Use a multi-process data loader, pinned memory and prefetching.
    - **No fusion**: a great many small elementwise kernels, dominated by launch overhead and memory traffic. Use `torch.compile`, FlashAttention and a fused optimizer (`fused=True`).
    - **Communication not overlapped**: with several cards, the all-reduce blocks the computation. Check DDP's bucketing and whether `no_sync` is used under gradient accumulation.
    - **Synchronisation points**: `.item()` and printing inside the training loop force GPU synchronisation, so do them less often.
    - Look at the timeline with a profiler (`torch.profiler` or Nsight Systems) and find the stretches where the GPU is idle, which is more reliable than guessing.

**3. More data.** Add three more classic novels (or any Chinese corpus you have) to the training set and retrain the tokenizer and the 3.3M model. What changes in the validation loss and the generated text?

??? success "The approach"
    With several times the data, the gap between the training and validation losses narrows noticeably at the same step count (it is no longer just memorising), and the larger model starts to show its advantage. This is scaling laws' "data and parameters grow together" made visible. With the styles mixed, the generated text picks up vocabulary from the different works, which also shows that the corpus's proportions directly shape the model's character.

## Summary {#小结}

- [x] Training on one token is about 6 times the parameter count in operations; convert the throughput into the achieved rate and the utilization, which is generally 30% to 50% for a Transformer on a GPU.
- [x] Low precision is only fast when the hardware supports it (bf16 is slower on a CPU); `torch.compile` fuses the scattered operations.
- [x] Gradient accumulation (each part's loss divided by the number of parts) is equivalent to a large batch and trades time for memory; DDP has each rank read different data and averages the gradients, matching a single process's large batch, with logging and checkpointing only on rank 0.
- [x] A larger model has a lower validation loss, but with too little data the returns diminish and the overfitting worsens; Chinchilla: about 20 tokens per parameter at compute-optimal.
- [x] Estimate the compute, the time and the memory before scaling up: DDP at the 1B scale, and at 7B the model states have to be partitioned with FSDP or ZeRO-3.
