import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, n = dist.get_rank(), dist.get_world_size()
checks = []


def try_op(name, fn):
    try:
        fn()
        checks.append((name, "可用"))
    except Exception as e:
        checks.append((name, type(e).__name__))


x = torch.ones(4 * n) * (rank + 1)
try_op("all_reduce", lambda: dist.all_reduce(x.clone()))
try_op("all_gather_into_tensor", lambda: dist.all_gather_into_tensor(torch.zeros(4 * n * n), x.clone()))
try_op("reduce_scatter_tensor", lambda: dist.reduce_scatter_tensor(torch.zeros(4), x.clone()))
try_op("all_to_all_single", lambda: dist.all_to_all_single(torch.zeros(4 * n), x.clone()))
try_op("broadcast", lambda: dist.broadcast(x.clone(), 0))
try_op("barrier", dist.barrier)
if rank == 0:
    print(f"后端 {dist.get_backend()}，{n} 个进程")
    for name, ok in checks:
        print(f"  {name:24s} {ok}")
dist.destroy_process_group()
