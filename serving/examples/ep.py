"""ep.py —— 专家并行（EP）+ DP Attention 的最小实现：两次 all-to-all（dispatch 与 combine）。

用法：torchrun --standalone --nproc-per-node 4 ep.py
每个 rank 处理自己的一批 token（DP Attention：注意力部分各 rank 处理不同的请求），
MoE 层的 E 个专家平均分给各 rank；token 按路由结果发给专家所在的 rank，算完再发回来。
"""

import os

import torch
import torch.distributed as dist

from moe import MoE

D, D_FF, N_EXPERTS, TOP_K = 64, 128, 16, 2


def all_to_all(x: torch.Tensor, send_counts: list[int]) -> tuple[torch.Tensor, list[int]]:
    """先交换"各发多少"，再按这个数量交换数据。"""
    world = dist.get_world_size()
    counts = torch.tensor(send_counts)
    recv = torch.empty(world, dtype=torch.long)
    dist.all_to_all_single(recv, counts, [1] * world, [1] * world)
    recv_counts = recv.tolist()
    out = torch.empty((sum(recv_counts),) + tuple(x.shape[1:]), dtype=x.dtype)
    dist.all_to_all_single(out, x.contiguous(), recv_counts, send_counts)
    return out, recv_counts


def ep_forward(moe: MoE, x: torch.Tensor, rank: int, world: int) -> tuple[torch.Tensor, int]:
    per_rank = N_EXPERTS // world
    weights, idx, _ = moe.route(x)                                     # 路由器在每个 rank 上都有一份
    flat_expert = idx.flatten()                                        # [N*k] 每个 (token, 槽位) 要去的专家
    flat_token = torch.arange(x.shape[0]).repeat_interleave(TOP_K)
    dest = flat_expert // per_rank                                     # 专家所在的 rank
    order = dest.argsort(stable=True)                                  # 按目标 rank 排好，才能按段发送
    send_counts = torch.bincount(dest, minlength=world).tolist()

    # dispatch：把 token 的隐藏状态和"要去哪个本地专家"发到专家所在的 rank
    recv_x, recv_counts = all_to_all(x[flat_token[order]], send_counts)
    recv_e, _ = all_to_all(flat_expert[order] - dest[order] * per_rank, send_counts)

    # 本地计算：按专家分组，每个本地专家对分到它的所有 token 做一次矩阵乘法
    y = torch.empty_like(recv_x)
    for e in range(per_rank):
        sel = (recv_e == e).nonzero().flatten()
        if len(sel):
            y[sel] = moe.experts[rank * per_rank + e](recv_x[sel])

    # combine：把结果发回 token 原来所在的 rank，再按路由权重加回原位置
    back, _ = all_to_all(y, recv_counts)
    out = torch.zeros_like(x)
    out.index_add_(0, flat_token[order], back * weights.flatten()[order, None])
    return out + sum(s(x) for s in moe.shared), recv_x.shape[0]


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)
    moe = MoE(D, D_FF, N_EXPERTS, TOP_K)                              # 所有 rank 用同一个种子，得到同一份完整权重
    torch.manual_seed(100 + rank)
    n_tokens = [96, 32, 64, 128][rank % 4]                             # 各 rank 的 batch 大小不同（DP）
    x = torch.randn(n_tokens, D)
    with torch.no_grad():
        out, n_received = ep_forward(moe, x, rank, world)

    # 在 rank 0 上汇总所有 token，用单进程的完整 MoE 计算作为参考（各 rank 大小不同，用对象收集）
    gathered = [None] * world
    dist.all_gather_object(gathered, (x, out, n_received))
    if rank == 0:
        xs, outs, loads = zip(*gathered)
        with torch.no_grad():
            ref = moe.forward_grouped(torch.cat(xs))
        err = (torch.cat(outs) - ref).abs().max().item()
        print(f"EP={world}：每个 rank {N_EXPERTS // world} 个专家；各 rank 的 token 数 {[len(t) for t in xs]}")
        print(f"各 rank 收到的 (token, 专家) 对数：{list(loads)}（总数 = token 数 × top-{TOP_K} = {sum(len(t) for t in xs) * TOP_K}）")
        print(f"与单进程完整 MoE 的最大误差：{err:.1e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
