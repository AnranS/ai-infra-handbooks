import torch
import torch.distributed as dist

from cp_common import attention, make_qkv

dist.init_process_group("gloo")
rank, P = dist.get_rank(), dist.get_world_size()
q, k, v = make_qkv()
S = q.shape[0]
ref, _ = attention(q, k, v)
C = S // P
mine = slice(rank * C, (rank + 1) * C)
q_l, kv = q[mine], torch.stack([k[mine], v[mine]])          # 本 rank 的 query 块和 KV 块

out, lse = None, None
src = rank                                                    # 当前手里的 KV 块来自哪个 rank（即序列里的第几块）
computed = 0
for step in range(P):
    if src <= rank:                                           # 因果：只有不晚于自己的 KV 块才有贡献
        o, l = attention(q_l, kv[0], kv[1], causal=True, q_offset=rank * C, k_offset=src * C)
        if out is None:
            out, lse = o, l
        else:                                                 # 用 log-sum-exp 合并两部分注意力（online softmax）
            new = torch.logaddexp(lse, l)
            out = out * torch.exp(lse - new).T[..., None] + o * torch.exp(l - new).T[..., None]
            lse = new
        computed += 1
    if step < P - 1:                                          # 把 KV 块传给右边，从左边接收下一块
        recv = torch.empty_like(kv)
        reqs = [dist.isend(kv, (rank + 1) % P), dist.irecv(recv, (rank - 1) % P)]
        for r in reqs:
            r.wait()
        kv, src = recv, (src - 1) % P

ok = torch.tensor([int(torch.allclose(out, ref[mine], atol=1e-5))])
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
counts = [None] * P
dist.all_gather_object(counts, computed)
if rank == 0:
    print(f"Ring Attention，{P} 个 rank：输出与单进程的因果注意力一致：", bool(ok.item()))
    print("每个 rank 实际计算的块数：", counts)
dist.destroy_process_group()
