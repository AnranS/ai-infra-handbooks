import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup():
    dist.init_process_group("gloo")
    return dist.get_rank(), dist.get_world_size()


def make_model(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(),
                               torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 8))


def data(n=64):
    g = torch.Generator().manual_seed(123)
    return torch.randn(n, 32, generator=g), torch.randint(0, 8, (n,), generator=g)


def local_slice(n, rank, world):
    return slice(rank * n // world, (rank + 1) * n // world)


def report(name, ok, detail=""):
    flag = torch.tensor([1 if ok else 0])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)                 # 所有 rank 都通过才算通过
    if dist.get_rank() == 0:
        print(f"{'PASS' if flag.item() else 'FAIL'}  {name}  {detail}", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if flag.item() else 1)
