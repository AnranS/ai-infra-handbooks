import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()


def show(title, t):
    """把每个 rank 上的结果收集到 rank 0 打印"""
    parts = [torch.empty_like(t) for _ in range(world)] if rank == 0 else None
    dist.gather(t, parts, dst=0)
    if rank == 0:
        print(title)
        for r, p in enumerate(parts):
            print(f"  rank {r}: {p.tolist()}")


x = torch.arange(4, dtype=torch.float32) + 10 * rank     # rank r 持有 [10r, 10r+1, 10r+2, 10r+3]
show("输入", x)

y = x.clone()
dist.all_reduce(y)                                        # 所有 rank 得到逐元素之和
show("all_reduce（求和）", y)

chunk = torch.empty(1)
dist.reduce_scatter_tensor(chunk, x)                      # 求和之后切成 world 份，rank r 拿第 r 份
show("reduce_scatter", chunk)

gathered = torch.empty(world)
dist.all_gather_into_tensor(gathered, chunk)              # 把每个 rank 的一份拼起来，所有 rank 都拿到全部
show("all_gather（接在 reduce_scatter 后面）", gathered)

out = torch.empty(4)
dist.all_to_all_single(out, x)                            # rank r 的第 p 个元素发给 rank p
show("all_to_all", out)

ok = torch.equal(gathered, y)
flag = torch.tensor([int(ok)])
dist.all_reduce(flag, op=dist.ReduceOp.MIN)
if rank == 0:
    print("reduce_scatter + all_gather == all_reduce：", bool(flag.item()))
dist.destroy_process_group()
