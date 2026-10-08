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
