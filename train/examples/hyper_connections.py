import torch

torch.manual_seed(0)
n, layers = 4, 60                                             # 4 条残差流（DeepSeek-V4 的 hc_mult），60 层


def sinkhorn(M, iters=20):
    """Sinkhorn-Knopp：交替把行和、列和归一化成 1，得到双随机矩阵（mHC 用 20 次迭代）"""
    for _ in range(iters):
        M = M / M.sum(dim=1, keepdim=True)
        M = M / M.sum(dim=0, keepdim=True)
    return M


def product_norm(make):
    P = torch.eye(n)
    norms = []
    for layer in range(1, layers + 1):
        P = make() @ P                                         # 残差流经过每一层的混合矩阵
        if layer in (10, 30, 60):
            norms.append(torch.linalg.matrix_norm(P, ord=2).item())
    return norms


# HC：混合矩阵不加约束（这里是单位阵加一点扰动，训练中学出来的矩阵不会恰好保持范数）
hc = product_norm(lambda: torch.eye(n) + 0.15 * torch.randn(n, n))
# mHC：先变成正矩阵，再用 Sinkhorn 投影成双随机矩阵（行和、列和都是 1）
mhc = product_norm(lambda: sinkhorn((torch.eye(n) + 0.15 * torch.randn(n, n)).exp()))
print("混合矩阵连乘之后的谱范数     10 层      30 层      60 层")
print("HC（不加约束）          " + "".join(f"{v:10.3g}" for v in hc))
print("mHC（双随机矩阵）        " + "".join(f"{v:10.3g}" for v in mhc))
M = sinkhorn((torch.eye(n) + 0.15 * torch.randn(n, n)).exp())
print(f"一个双随机矩阵：行和 {[round(v, 3) for v in M.sum(1).tolist()]}，谱范数 {torch.linalg.matrix_norm(M, ord=2):.3f}")
