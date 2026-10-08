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
