import torch
import torch.distributed as dist
from muon import newton_schulz

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)                                        # 每个 rank 造出同样的"已经 all-reduce 过的"梯度
shapes = [(256, 256), (256, 1024), (1024, 256), (256, 768)]  # 注意力、MLP 的几种矩阵
grads = [torch.randn(s) @ torch.diag(torch.logspace(0, -2, s[1])) for s in shapes]


def rows(t):                                                # FSDP2 式的切分：每个参数按第 0 维切，rank 拿连续的若干行
    n = t.shape[0] // world
    return t[rank * n:(rank + 1) * n]


def gather_rows(local):
    parts = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(parts, local)
    return torch.cat(parts)


ref = [newton_schulz(g) for g in grads]                     # 单卡 Muon 的更新（不带动量，只看正交化这一步）

# ① 每个 rank 只对自己那几行做正交化：矩阵被拆开了，结果是错的
local = [newton_schulz(rows(g)) for g in grads]
err1 = max(((gather_rows(u) - r).norm() / r.norm()).item() for u, r in zip(local, ref))

# ② 先 all-gather 出完整矩阵，每个 rank 都做一遍正交化，再取自己的行（Moonshot 的 Distributed Muon 思路）
calls2 = 0
out2 = []
for g in grads:
    full = gather_rows(rows(g))
    out2.append(rows(newton_schulz(full)))
    calls2 += 1
err2 = max((gather_rows(u) - r).abs().max().item() for u, r in zip(out2, ref))

# ③ 每个矩阵交给一个 rank：gather 到那个 rank，只在那里正交化，再 scatter 回各个 rank
calls3 = 0
out3 = []
for i, g in enumerate(grads):
    owner = i % world
    parts = [torch.empty_like(rows(g)) for _ in range(world)] if rank == owner else None
    dist.gather(rows(g), parts, dst=owner)
    pieces = None
    if rank == owner:
        pieces = list(newton_schulz(torch.cat(parts)).chunk(world))
        calls3 += 1
    mine = torch.empty_like(rows(g))
    dist.scatter(mine, pieces, src=owner)
    out3.append(mine)
err3 = max((gather_rows(u) - r).abs().max().item() for u, r in zip(out3, ref))
calls = torch.tensor([calls3])
dist.all_reduce(calls, op=dist.ReduceOp.MAX)

if rank == 0:
    numel = sum(a * b for a, b in shapes)
    print(f"{world} 个 rank，{len(shapes)} 个矩阵，共 {numel / 1e6:.2f}M 个参数")
    print(f"① 只对本地的行正交化：与单卡的相对误差 {err1:.0%}")
    print(f"② all-gather 后各自正交化：与单卡的最大差 {err2:.1e}，每个 rank 做 {calls2} 次 Newton-Schulz")
    print(f"③ 每个矩阵交给一个 rank：与单卡的最大差 {err3:.1e}，每个 rank 最多做 {calls.item()} 次 Newton-Schulz")
dist.destroy_process_group()
