# 从零训练（三）：训得更快、更大

<p class="lead">小模型已经训出来了。这一章回答接下来的问题：训练一步的时间花在哪里、算力用上了多少；显存放不下更大的 batch 怎么办；怎样用多张卡一起训练、结果还和单卡一致；模型做大一些效果能好多少；以及真正放到 GPU 上时，一个 1B 模型要多少卡、多少小时。每个问题都先在 CPU 上用同一个模型实测，再推到 GPU 和大模型上，并指向这本手册里讲同一件事的章节。</p>

!!! note "这一章跑什么"
    整章都是实验，不产出主线要用的文件，跳过也不影响后面：`speed.py` 测吞吐与 MFU（约 20 秒）、
    `grad_accum.py` 验证梯度累积、`ddp_train.py` 开两个进程跑 DDP、`scaling.py` 训练四个尺寸（约 2.6 分钟）、
    `budget.py` 估算 GPU 上的时间与成本；`train_gpu.py` 需要 GPU，只看不跑。

!!! question "自测：能答上来就可以跳过本章"
    1. 训练一个 token 大约需要多少次浮点运算？MFU 是怎么算的？
    2. 梯度累积为什么和大 batch 等价？loss 要怎么缩放？
    3. DDP 训练时，每个 rank 的数据有什么不同？checkpoint 由谁来存？
    4. 同样的数据量，模型越大越好吗？Chinchilla 的"每个参数 20 个 token"是什么意思？
    5. 一个 1.3B 的模型按计算最优的数据量训练，8 张 H100 大约要多久？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 约 $6N$ 次（前向 $2N$、反向 $4N$）。MFU = 每秒处理的 token 数 × $6N$ ÷ 硬件的峰值算力；GPU 上训练 Transformer 一般在 30%～50%。
    2. 梯度是对每个样本求梯度再平均，把一个大 batch 拆成几份分别反向、梯度累加起来，和一次性算整个 batch 的梯度相同。每一份的 loss 要除以份数，累加起来才是平均的梯度。
    3. 每个 rank 读不同的数据（按 rank 编号错开，或者用 DistributedSampler）；梯度 all-reduce 取平均后参数保持一致，所以只需要 rank 0 写日志和保存 checkpoint。
    4. 不一定：数据量固定时，模型大到一定程度，收益递减、过拟合加重。Chinchilla 说的是：给定算力预算时，损失最低的配置大约是每个参数训练 20 个 token，参数量和数据量同比例增长。
    5. 按 20 token/参数是 26B token，约 $6 \times 1.3 \times 10^9 \times 2.6 \times 10^{10} \approx 2 \times 10^{20}$ 次运算；8 张 H100 按合理的 MFU 大约 18 小时（本章估算为 17.8 小时）。

## 先测量：时间花在哪里

训练一个 token 的计算量约为 **6 × 参数量**：前向每个参数一次乘加（2 次浮点运算），反向算输入的梯度和权重的梯度各一次，一共 3 倍。注意力里 $QK^\top$ 和 $PV$ 的计算不在参数里，要另外加上（上下文短时占比很小）。用它把"每秒处理多少 token"换算成"实际用上了多少算力"：

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
            t0 = time.perf_counter()                          # 前几步是预热（torch.compile 在这里编译）
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

下面是在一台服务器的 2 个 CPU 核上的结果（耗时每次都不一样，所以不逐行核对）：

```text
fp32                   每步  224.2 ms，    9136 token/s，实际算力 106.0 GFLOPS
bf16 autocast          每步  619.4 ms，    3306 token/s，实际算力  38.4 GFLOPS
fp32 + torch.compile   每步  184.4 ms，   11108 token/s，实际算力 128.9 GFLOPS
```

- **CPU 上 bf16 反而慢了近 3 倍**：这颗 CPU 没有 bf16 的矩阵运算指令，autocast 只是多了类型转换。低精度只有在硬件支持时才快——GPU 的 Tensor Core 上，bf16 的算力是 fp32 的十几倍，混合精度才是训练的默认选择（见[混合精度](train://practice/mixed-precision/)）；
- **torch.compile 快了约 20%**：把 RMSNorm、SwiGLU、RoPE 这些零碎的逐元素算子融合成少数几个 kernel，少读写几遍内存（原理见 CUDA 手册的 [torch.compile](cuda://framework/compile/)）。GPU 上小模型的收益通常更大，因为 kernel 启动开销更显眼；
- **MFU**（Model FLOPs Utilization）= 实际算力 / 硬件峰值。GPU 上训练 Transformer 的 MFU 一般在 30%～50%，低于 20% 说明有明显的浪费：batch 太小、数据加载跟不上、通信没有和计算重叠、太多小 kernel。

## 显存不够：梯度累积

![图：梯度累积——几个 micro-batch 的梯度累加后再更新一次](assets/figures/grad-accum.svg){.aig-svg}

batch 越大，梯度的噪声越小，但激活占的显存也越大。**梯度累积**把一个大 batch 拆成几份依次算，梯度加起来再更新一次，数学上完全等价——前提是每份的 loss 要除以份数：

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
        (loss / micro_batches).backward()                     # 每个小 batch 的 loss 除以份数：梯度累加起来就是大 batch 的平均
    return torch.cat([p.grad.flatten() for p in model.parameters()])


full = grads(1)
for k in (2, 4, 8):
    diff = (grads(k) - full).abs().max().item()
    print(f"32 条分成 {k} 份累加：与一次算 32 条的梯度最大差小于 1e-6：{diff < 1e-6}（梯度最大值 {full.abs().max():.2f}）")
```

```text title="输出"
32 条分成 2 份累加：与一次算 32 条的梯度最大差小于 1e-6：True（梯度最大值 0.06）
32 条分成 4 份累加：与一次算 32 条的梯度最大差小于 1e-6：True（梯度最大值 0.06）
32 条分成 8 份累加：与一次算 32 条的梯度最大差小于 1e-6：True（梯度最大值 0.06）
```

代价是时间：拆成 $k$ 份，就要串行地跑 $k$ 次前向和反向。它和"激活重计算"是省显存的两个基本办法（显存账本见[总论](train://basics/overview/)）。注意 BatchNorm 这类跨样本的算子会让两者不再等价，Transformer 用的 LayerNorm / RMSNorm 没有这个问题。

## 多张卡：DDP

数据并行（DDP）：每张卡放一份完整的模型，各自处理不同的数据，反向时对梯度取平均（all-reduce），于是每张卡的参数始终相同。和单进程、每步用所有 rank 的数据拼成的大 batch 比较：

```python title="ddp_train.py" torchrun="2"
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

dist.init_process_group("gloo")                               # GPU 上换成 "nccl"，模型和数据放到 cuda:{local_rank}
rank, world = dist.get_rank(), dist.get_world_size()
data = torch.load("tokens.pt")["train"].long()
cfg = GPTConfig(d_model=64, n_head=2, n_layer=2, seq_len=64)
PER_RANK, STEPS = 8, 30


def batch(step, r):
    """第 step 步、第 r 个 rank 的数据：每个 rank 拿不同的片段（真实训练里是数据集的不同分片）"""
    g = torch.Generator().manual_seed(1000 * step + r)
    i = torch.randint(0, len(data) - 65, (PER_RANK,), generator=g)
    return torch.stack([data[j:j + 64] for j in i]), torch.stack([data[j + 1:j + 65] for j in i])


torch.manual_seed(0)                                           # 每个 rank 同样的初始化（DDP 构造时还会从 rank 0 广播一次）
model = DDP(GPT(cfg))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
for step in range(STEPS):
    x, y = batch(step, rank)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()                                            # 反向时按桶 all-reduce 梯度（取平均）
    opt.step()

if rank == 0:                                                  # 对照：单进程，每步把所有 rank 的数据拼成一个大 batch
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
    torch.save(model.module.state_dict(), "ddp_model.pt")   # 只在 rank 0 存 checkpoint；存的是 .module，不带 DDP 的包装
dist.destroy_process_group()
```

```text title="输出"
2 个进程，每个每步 8 条：训练 30 步后与单进程（每步 16 条）的参数最大差小于 1e-4：True
```

- **每个 rank 的数据必须不同**，否则多张卡算的是同一份梯度，等于白算。真实的数据加载器按 rank 切分数据集（`DistributedSampler`、按分片读取）；
- **全局 batch = 每卡的 batch × 卡数 × 梯度累积的份数**。加卡时要保持全局 batch 不变（或者相应地调学习率），否则训练的动力学就变了；
- **日志、验证和 checkpoint 只在 rank 0 做**，存的是 `model.module`，不带 DDP 的包装；
- 差异来自浮点累加的顺序（all-reduce 的求和顺序和单进程不同）。DDP 怎样把梯度分桶、和反向重叠，见[数据并行与 DDP](train://data/ddp/)；模型大到一张卡放不下完整的参数和优化器状态时，就要用 [ZeRO 与 FSDP](train://data/zero-fsdp/) 把它们切开。

## 做大模型：效果能好多少

同样的数据、同样的 400 步，训练四个大小不同的模型：

```python title="scaling.py"
import math

import torch

from model import GPT, GPTConfig

torch.set_num_threads(8)                                       # 这个实验要训 4 个模型，多用几个线程
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

```text title="输出"
  d  层数  参数量    验证 loss：第 100    200    300    400 步   最后的训练 loss
 64    2   0.62M         6.85   6.63   6.52   6.49       5.83
 96    3   1.12M         6.79   6.52   6.39   6.34       5.65
128    4   1.80M         6.74   6.47   6.33   6.26       5.52
192    4   3.34M         6.70   6.44   6.28   6.22       5.41
```

- **模型越大，验证 loss 越低**，而且在每个阶段都是如此；但收益递减：参数翻了 5 倍，验证 loss 只低了 0.27；
- **训练 loss 与验证 loss 的差距随模型变大而变大**：大模型更会"背书"。我们的数据只有 41 万个 token，四个模型都已经处在"数据不够"的区间；
- 这正是**规模定律**（scaling laws）研究的问题：给定计算预算 $C \approx 6ND$，参数量 $N$ 和数据量 $D$ 该怎么分？Chinchilla 的结论是两者应该同比例增长，大约**每个参数 20 个 token**。我们 1.8M 参数的模型按这个比例需要 3600 万个 token，是现有数据的 90 倍——在小数据上继续加大模型，收益会越来越小。真实的大模型往往训练得远超 20 个 token / 参数（比如 7B 模型训练几万亿 token），因为推理的成本和参数量成正比，多花训练算力换一个更小、更强的模型是划算的。

## 放到 GPU 上

GPU 版的训练脚本：bf16 自动混合精度、`torch.compile`、DDP 和梯度累积。它需要 NVIDIA GPU，这里只做语法检查：

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
torch.backends.cuda.matmul.allow_tf32 = True                  # 剩下的 fp32 矩阵乘也用 Tensor Core（TF32）

cfg = GPTConfig(d_model=512, n_layer=8, n_head=8, seq_len=512)   # 约 29M 参数
MICRO, ACCUM, STEPS, LR, WARMUP = 32, 4, 2000, 1e-3, 100
global_batch = MICRO * ACCUM * world                         # 每步的全局 batch（条数）
data = torch.load("tokens.pt")                                # 换成更大的语料才能真正用上 GPU（见正文）
train_data = data["train"].long()
gen = torch.Generator().manual_seed(1000 + rank)              # 每个 rank 取不同的数据


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
peak = 989e12                                                 # H100 bf16 稠密算力；A100 是 312e12

t0 = time.perf_counter()
for step in range(STEPS):
    lr = LR * min(1.0, (step + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
    for group in opt.param_groups:
        group["lr"] = lr
    for micro in range(ACCUM):
        x, y = get_batch()
        sync = (not distributed) or micro == ACCUM - 1       # 只在最后一个 micro-batch 同步梯度
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

几个和 CPU 版不同的地方：

- `torch.autocast("cuda", dtype=torch.bfloat16)` 让矩阵乘用 bf16，权重和优化器状态仍是 fp32（主权重）；`allow_tf32` 让剩下的 fp32 矩阵乘也用上 Tensor Core；
- 梯度累积时，前几个 micro-batch 用 `model.no_sync()` 跳过梯度同步，只在最后一个 micro-batch 做 all-reduce，通信量减少到 $1/k$；
- 数据用 `pin_memory` + `non_blocking` 异步拷到 GPU，和计算重叠；
- 每 50 步打印吞吐、MFU 和峰值显存，这三个数是调优的起点；
- 本书的语料只有 41 万个 token，这个 29M 参数的模型在 8 张卡上一步就读完好几遍——要真正用上 GPU，得换一份大得多的语料（比如开源中文预训练语料的一个子集）。

最后，按计算最优的数据量（20 个 token / 参数）估算把模型做大需要的资源：

```python title="budget.py"
# 把教程里的模型放大：算力、时间和显存的估算（H100：bf16 稠密算力 989 TFLOPS，80 GB）
PEAK, MFU = 989e12, 0.4


def estimate(name, n_params, gpus, d_model, n_layer, seq_len, micro_batch):
    tokens = 20 * n_params                                      # Chinchilla：计算最优时约 20 个 token / 参数
    flops = 6 * n_params * tokens                               # 前向 + 反向 ≈ 6 × 参数量 × token 数
    hours = flops / (gpus * PEAK * MFU) / 3600
    states = 16 * n_params                                      # bf16 权重和梯度 + fp32 主权重 + Adam 的两个矩
    acts = 34 * seq_len * micro_batch * d_model * n_layer       # 每层约 34·s·b·h 字节的激活（bf16、FlashAttention、不重算）
    print(f"{name:14s} {tokens / 1e9:6.1f}B token  {flops:8.1e} FLOP  {gpus:3d} 卡 {hours:6.1f} 小时   "
          f"模型状态 DDP {states / 1e9:5.1f} GB / FSDP {states / gpus / 1e9:5.2f} GB，激活 {acts / 1e9:5.1f} GB")


print("模型              数据量        总计算量        时间（MFU 40%）       每卡显存")
estimate("124M（GPT-2）", 124e6, 1, 768, 12, 1024, 16)
estimate("1.3B", 1.3e9, 8, 2048, 24, 2048, 8)
estimate("7B", 7e9, 64, 4096, 32, 4096, 2)
```

```text title="输出"
模型              数据量        总计算量        时间（MFU 40%）       每卡显存
124M（GPT-2）       2.5B token   1.8e+18 FLOP    1 卡    1.3 小时   模型状态 DDP   2.0 GB / FSDP  1.98 GB，激活   5.1 GB
1.3B             26.0B token   2.0e+20 FLOP    8 卡   17.8 小时   模型状态 DDP  20.8 GB / FSDP  2.60 GB，激活  27.4 GB
7B              140.0B token   5.9e+21 FLOP   64 卡   64.5 小时   模型状态 DDP 112.0 GB / FSDP  1.75 GB，激活  36.5 GB
```

- 124M 的 GPT-2 规模，一张 H100 一两个小时就能训完，是个人复现的好起点；
- 1.3B 用 8 卡 DDP 不到一天，每卡要放下 20.8 GB 的模型状态和 27 GB 的激活，加上输出层的 logits 和各种临时缓冲，micro-batch 已经不能再翻倍，否则要开激活重计算；
- 7B 的模型状态（112 GB）一张卡根本放不下，必须用 FSDP / ZeRO-3 切到各张卡上（每卡 1.75 GB），激活也要靠重计算或者更小的 micro-batch 控制——从这里开始，就进入这本手册后面各章讲的[并行策略](train://practice/strategy/)了。

!!! interview "怎么讲清楚"
    讲"给你 8 张 H100，训练一个 1B 左右的模型，怎么规划"：**算预算**（$C \approx 6ND$，按 20 token / 参数取 $D \approx 20B$；MFU 按 40% 估，约十个小时）→ **算显存**（模型状态 16 字节 / 参数约 16 GB，加激活；放得下就用 DDP，放不下用 FSDP 或激活重计算）→ **定 batch**（全局 batch 几百万 token，用每卡 batch × 卡数 × 累积份数凑出来）→ **先小后大**（在小模型上把数据、分词器、训练循环、续训、评估跑通，扫学习率，再放大）→ **监控**（loss、梯度范数、吞吐、MFU、显存，定期看生成样例和评测）。

## 练习

**1. 扫学习率。** 用 `scaling.py` 的写法，对 1.8M 的模型试 1e-3、3e-3、1e-2 三个学习率，哪个最好？换成 3.3M 的模型，最好的学习率会变吗？

??? success "参考思路"
    一般会看到一个"U 形"：学习率太小，400 步里走不了多远；太大，loss 更抖、甚至发散。模型变宽时最优学习率通常会变小。真实训练里，扫学习率要在小模型上做，然后按经验规律（或者 μP 这样的参数化方法）迁移到大模型，因为在大模型上扫一遍的成本太高。

**2. MFU 为什么上不去？** 一个 GPU 训练任务的 MFU 只有 15%，列出你会检查的几件事。

??? success "参考答案"
    - **batch 太小**：矩阵乘太小，用不满 Tensor Core。增大 micro-batch 或序列长度；
    - **数据加载**：GPU 在等 CPU 准备数据。用多进程的数据加载器、pin_memory、预取；
    - **没有融合**：大量逐元素的小 kernel，启动开销和显存读写占主导。用 `torch.compile`、FlashAttention、融合的优化器（`fused=True`）；
    - **通信没有重叠**：多卡时 all-reduce 阻塞了计算。检查 DDP 的分桶、梯度累积时是否用了 `no_sync`；
    - **同步点**：训练循环里有 `.item()`、打印这类强制 GPU 同步的操作，降低频率；
    - 用 profiler（`torch.profiler` 或 Nsight Systems）看时间线，找到 GPU 空闲的时间段，比猜更可靠。

**3. 扩大数据。** 把另外三部名著（或任何你有的中文语料）加进训练集，重新训练分词器和 3.3M 的模型。验证 loss 和生成的文字有什么变化？

??? success "参考思路"
    数据量增加几倍之后，同样的步数下训练 loss 与验证 loss 的差距会明显缩小（不再只是"背书"），更大的模型也开始体现出优势——这就是规模定律里"数据和参数要一起增长"的直观体现。文风混合之后，生成的文字会带上不同作品的用词，这也说明数据的配比会直接影响模型的"性格"。

## 小结

- [x] 训练一个 token ≈ 6 × 参数量次浮点运算；用吞吐换算实际算力和 MFU，GPU 上训练 Transformer 一般在 30%～50%。
- [x] 低精度只有硬件支持时才快（CPU 上 bf16 反而慢）；`torch.compile` 融合零碎的算子。
- [x] 梯度累积（每份 loss 除以份数）和大 batch 等价，用时间换显存；DDP 每个 rank 读不同的数据、对梯度取平均，结果和单进程的大 batch 一致，日志和 checkpoint 只在 rank 0。
- [x] 模型越大验证 loss 越低，但数据不够时收益递减、过拟合加重；Chinchilla：计算最优时每个参数约 20 个 token。
- [x] 放大之前先估算算力、时间和显存：1B 级别用 DDP，7B 级别的模型状态必须用 FSDP / ZeRO-3 切开。
