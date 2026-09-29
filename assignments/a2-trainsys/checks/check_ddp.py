"""torchrun --nproc-per-node 2 checks/check_ddp.py"""
import torch
import torch.distributed as dist

from common import data, local_slice, make_model, report, setup

calls, in_backward = [], [False]
_all_reduce = dist.all_reduce


def spy(tensor, *args, **kwargs):                              # 记录每次 all-reduce 是否发生在反向过程中、是否异步
    calls.append((in_backward[0], kwargs.get("async_op", False)))
    return _all_reduce(tensor, *args, **kwargs)


dist.all_reduce = spy
from trainsys.ddp import BucketedDDP  # noqa: E402

rank, world = setup()
X, Y = data()
local = local_slice(len(X), rank, world)
ref = make_model(0)
model = BucketedDDP(make_model(rank), bucket_bytes=32 * 1024)   # 各 rank 初始参数不同：构造时必须以 rank 0 为准
opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()
for _ in range(3):
    opt_ref.zero_grad()
    loss_fn(ref(X), Y).backward()
    opt_ref.step()
    opt.zero_grad()
    in_backward[0] = True
    loss_fn(model(X[local]), Y[local]).backward()
    in_backward[0] = False
    model.finish_gradient_sync()
    opt.step()
same = all(torch.allclose(p, q, atol=1e-5) for p, q in zip(model.parameters(), ref.parameters()))
overlap = any(b and a for b, a in calls)
report("BucketedDDP", same and overlap, f"与单进程一致：{same}；反向过程中发起了异步 all-reduce：{overlap}")
