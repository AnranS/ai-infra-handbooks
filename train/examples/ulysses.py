import torch
import torch.distributed as dist

from cp_common import attention, make_qkv

dist.init_process_group("gloo")
rank, P = dist.get_rank(), dist.get_world_size()
q, k, v = make_qkv()                        # 完整的 [S, H, D]，用来算参照结果和切出本 rank 的输入
S, H, D = q.shape
ref, _ = attention(q, k, v)
rows = slice(rank * S // P, (rank + 1) * S // P)


def seq_to_head(x):
    """[S/P, H, D]（本 rank 持有一段序列的全部头）→ [S, H/P, D]（全部序列的一部分头）"""
    x = x.reshape(S // P, P, H // P, D).transpose(0, 1).contiguous()   # 按目标 rank 分组：第 p 组是第 p 份头
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)                                    # 第 p 组发给 rank p；收到的第 r 组来自 rank r 的那段序列
    return out.reshape(S, H // P, D)


def head_to_seq(x):
    """[S, H/P, D] → [S/P, H, D]：上面的逆操作"""
    x = x.reshape(P, S // P, H // P, D).contiguous()                   # 第 p 组是第 p 段序列
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)
    return out.transpose(0, 1).reshape(S // P, H, D)


ql, kl, vl = seq_to_head(q[rows]), seq_to_head(k[rows]), seq_to_head(v[rows])
out_local, _ = attention(ql, kl, vl)       # 本 rank 对自己的 H/P 个头做完整序列的因果注意力
out = head_to_seq(out_local)
ok = torch.tensor([int(torch.allclose(out, ref[rows], atol=1e-5))])
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"Ulysses，{P} 个 rank：每个 rank 输入 {tuple(q[rows].shape)}，注意力时 {tuple(ql.shape)}")
    print("输出与单进程的因果注意力一致：", bool(ok.item()))
dist.destroy_process_group()
