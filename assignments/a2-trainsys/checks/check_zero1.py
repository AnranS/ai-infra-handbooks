"""torchrun --nproc-per-node 4 checks/check_zero1.py"""
import torch
import torch.distributed as dist

from common import data, local_slice, make_model, report, setup
from trainsys.zero1 import ZeRO1

rank, world = setup()
X, Y = data()
local = local_slice(len(X), rank, world)
ref, model = make_model(0), make_model(0)
kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
opt_ref = torch.optim.AdamW(ref.parameters(), **kw)
opt = ZeRO1(list(model.parameters()), **kw)
loss_fn = torch.nn.CrossEntropyLoss()
for _ in range(3):
    opt_ref.zero_grad()
    loss_fn(ref(X), Y).backward()
    opt_ref.step()
    opt.zero_grad()
    loss_fn(model(X[local]), Y[local]).backward()
    for p in model.parameters():                                # 梯度求平均（真实系统里由 DDP 完成）
        dist.all_reduce(p.grad)
        p.grad /= world
    opt.step()
same = all(torch.allclose(p, q, atol=1e-4) for p, q in zip(model.parameters(), ref.parameters()))   # AdamW 会放大求和顺序带来的微小差异
full = 2 * 4 * sum(p.numel() for p in model.parameters())      # 不切分时：fp32 的一阶矩 + 二阶矩
mine = opt.state_bytes()
total = torch.tensor([mine])
dist.all_reduce(total)
sharded = mine <= full / world * 1.1 + 4096 and total.item() >= full * 0.9
report("ZeRO1", same and sharded, f"与单进程 AdamW 一致：{same}；本卡状态 {mine / 1024:.0f} KiB，不切分 {full / 1024:.0f} KiB")
