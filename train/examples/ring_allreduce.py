import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, n = dist.get_rank(), dist.get_world_size()
right, left = (rank + 1) % n, (rank - 1) % n


def ring_all_reduce(t):
    """环形 all-reduce：先 n-1 步 reduce-scatter，再 n-1 步 all-gather。返回本 rank 发送的字节数。"""
    chunks = list(t.chunk(n))                      # 视图：直接在 t 上原地累加
    sent = 0
    for step in range(n - 1):                      # reduce-scatter：第 step 步把块 (rank - step) 发给右边
        send_idx, recv_idx = (rank - step) % n, (rank - step - 1) % n
        buf = torch.empty_like(chunks[recv_idx])
        reqs = [dist.isend(chunks[send_idx].contiguous(), right), dist.irecv(buf, left)]
        for r in reqs:
            r.wait()
        chunks[recv_idx] += buf
        sent += chunks[send_idx].numel() * t.element_size()
    for step in range(n - 1):                      # all-gather：把已经求好和的块沿环传一圈
        send_idx, recv_idx = (rank + 1 - step) % n, (rank - step) % n
        buf = torch.empty_like(chunks[recv_idx])
        reqs = [dist.isend(chunks[send_idx].contiguous(), right), dist.irecv(buf, left)]
        for r in reqs:
            r.wait()
        chunks[recv_idx].copy_(buf)
        sent += chunks[send_idx].numel() * t.element_size()
    return sent


torch.manual_seed(rank)
x = torch.randn(1024)
ref = x.clone()
dist.all_reduce(ref)
sent = ring_all_reduce(x)
ok = torch.tensor([int(torch.allclose(x, ref, atol=1e-5))])
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
sizes = [torch.zeros(1, dtype=torch.long) for _ in range(n)]
dist.all_gather(sizes, torch.tensor([sent]))
if rank == 0:
    total = x.numel() * x.element_size()
    print(f"{n} 个 rank，每个 rank 的数据 {total} 字节")
    print("与 dist.all_reduce 结果一致：", bool(ok.item()))
    print("每个 rank 发送的字节：", [int(s.item()) for s in sizes], f"= 2(n-1)/n × {total} = {2 * (n - 1) * total // n}")
dist.destroy_process_group()
