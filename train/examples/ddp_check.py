import torch
import torch.distributed as dist

from my_ddp import MyDDP

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(),
                               torch.nn.Linear(256, 10))


torch.manual_seed(123)
X, Y = torch.randn(32, 64), torch.randint(0, 10, (32,))   # 全局 batch：32 条样本
local = slice(rank * 32 // world, (rank + 1) * 32 // world)   # 每个 rank 拿自己的一份

ref = make_model()                                          # 单进程参照：用完整 batch 训练
ddp = MyDDP(make_model())
official = torch.nn.parallel.DistributedDataParallel(make_model())   # PyTorch 自带的 DDP，作为对照
opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
opt = torch.optim.SGD(ddp.parameters(), lr=0.1)
opt_off = torch.optim.SGD(official.parameters(), lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

for step in range(3):
    opt_ref.zero_grad()
    loss_fn(ref(X), Y).backward()
    opt_ref.step()

    opt.zero_grad()
    loss = loss_fn(ddp(X[local]), Y[local])                # 每个 rank 只算自己那份数据的平均损失
    loss.backward()
    ddp.finish_gradient_sync()                              # 梯度求平均：等价于对全局 batch 求平均
    opt.step()

    opt_off.zero_grad()
    loss_fn(official(X[local]), Y[local]).backward()        # 官方 DDP 在反向里自动同步
    opt_off.step()

def max_diff(m):
    return max((a - b).abs().max().item() for a, b in zip(ref.parameters(), m.parameters()))


if rank == 0:
    print(f"{world} 个 rank，{len(ddp.buckets)} 个梯度桶，每步在反向过程中发起 {ddp.launched_in_backward // 3} 次 all-reduce")
    print("训练 3 步后，自己写的 DDP 与单进程一致：", max_diff(ddp.module) < 1e-6)
    print("训练 3 步后，官方 DDP 与单进程一致：", max_diff(official.module) < 1e-6)
dist.destroy_process_group()
