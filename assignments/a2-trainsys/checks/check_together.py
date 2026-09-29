"""torchrun --nproc-per-node 2 checks/check_together.py：三者组合在一起训练，与单进程一致"""
import torch

from common import data, local_slice, report, setup
from trainsys.ddp import BucketedDDP
from trainsys.recompute import checkpoint
from trainsys.zero1 import ZeRO1


class Net(torch.nn.Module):
    def __init__(self, seed, ckpt):
        super().__init__()
        torch.manual_seed(seed)
        self.inp = torch.nn.Linear(32, 128)
        self.blocks = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.LayerNorm(128), torch.nn.Linear(128, 256), torch.nn.GELU(),
                                                              torch.nn.Linear(256, 128)) for _ in range(3))
        self.out = torch.nn.Linear(128, 8)
        self.ckpt = ckpt

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + (checkpoint(b, h) if self.ckpt else b(h))
        return self.out(h)


rank, world = setup()
X, Y = data()
local = local_slice(len(X), rank, world)
kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
ref = Net(0, ckpt=False)
opt_ref = torch.optim.AdamW(ref.parameters(), **kw)
model = BucketedDDP(Net(rank, ckpt=True), bucket_bytes=32 * 1024)
opt = ZeRO1(list(model.parameters()), **kw)
loss_fn = torch.nn.CrossEntropyLoss()
for _ in range(3):
    opt_ref.zero_grad()
    loss_fn(ref(X), Y).backward()
    opt_ref.step()
    opt.zero_grad()
    loss_fn(model(X[local]), Y[local]).backward()
    model.finish_gradient_sync()
    opt.step()
same = all(torch.allclose(p, q, atol=1e-4) for p, q in zip(model.parameters(), ref.parameters()))   # AdamW 会放大求和顺序带来的微小差异
report("DDP + ZeRO-1 + 重计算", same, f"与单进程一致：{same}")
