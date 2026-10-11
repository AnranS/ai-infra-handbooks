"""Gated DeltaNet 的分块并行版：块内用几次矩阵乘一次算完，块与块之间只传一张状态表 S。

逐 token 递推一步只做几个小矩阵运算，T 个 token 就是 T 次 Python 循环，GPU 完全吃不饱。
把序列切成 64 个 token 一块：块内 64 次"衰减 + 写入"合起来是一个下三角线性方程组（WY 表示），
用一次三角求解得到每个 token 真正写入的差值 u；然后块与块之间像递推一样传 S。结果和逐 token 版一样，误差在 1e-5 量级。
"""
import torch
import torch.nn.functional as F

from qwen.norm import l2norm


def chunked_delta_rule(q, k, v, g, beta, S=None, chunk=64):
    """和 delta_rule 同样的输入输出：q, k: [B, T, 头, dk]，v: [B, T, 头, dv]，g、beta: [B, T, 头]；S 是初始的记忆表"""
    B, T, n_head, dk = q.shape
    dv, dtype = v.shape[-1], v.dtype
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))                   # 记忆表和中间量一律 fp32（bf16 攒不住 64 步的累加）
    q, k = l2norm(q), l2norm(k)
    q = q / dk ** 0.5
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))                               # [B, 头, T, d]
    g, beta = g.transpose(1, 2), beta.transpose(1, 2)                               # [B, 头, T]
    pad = (chunk - T % chunk) % chunk                                               # 补到 chunk 的整数倍
    q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = (T + pad) // chunk
    q, k, v = (t.reshape(B, n_head, n, chunk, -1) for t in (q, k, v))               # [B, 头, 块数, 块长, d]
    g, beta = g.reshape(B, n_head, n, chunk), beta.reshape(B, n_head, n, chunk)

    G = g.cumsum(-1)                                                                # 块内累计衰减（对数）：从块首到第 t 个 token 一共衰减了多少
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    decay = (G[..., :, None] - G[..., None, :]).masked_fill(upper, float("-inf")).exp()   # decay[i, j] = 从 j 到 i 的衰减（j > i 时为 0）
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    A = (k_beta @ k.transpose(-1, -2)) * decay                                      # 块内每对 token 之间"写入互相抵消"的系数
    u = torch.linalg.solve_triangular(A, v_beta, upper=False, unitriangular=True)   # 解下三角方程组：每个 token 真正写入的差值（还没减去块首状态的预测）
    w = torch.linalg.solve_triangular(A, k_beta * G.exp()[..., None], upper=False, unitriangular=True)   # 用来从块首状态里读预测的系数
    attn = (q @ k.transpose(-1, -2)) * decay                                        # 块内 q 对块内 k 的"注意力"
    q_decay = q * G.exp()[..., None]                                                # 读块首状态时，q 要带上从块首到自己的衰减
    k_decay = k * (G[..., -1:] - G).exp()[..., None]                                # 写入状态时，k 要带上从自己到块尾的衰减
    chunk_decay = G[..., -1].exp()                                                  # 整块的总衰减

    if S is None:
        S = q.new_zeros(B, n_head, dk, dv)
    out = torch.zeros_like(v)
    for i in range(n):                                                              # 块间：和递推一样只传 S
        u_i = u[:, :, i] - w[:, :, i] @ S                                           # 减掉块首状态已经能预测的部分，得到真正的 delta
        out[:, :, i] = q_decay[:, :, i] @ S + attn[:, :, i] @ u_i                   # 输出 = 读块首状态 + 块内读新写入
        S = S * chunk_decay[:, :, i, None, None] + k_decay[:, :, i].transpose(-1, -2) @ u_i   # 整块衰减一次，再把块内的写入一次加上
    out = out.reshape(B, n_head, n * chunk, dv)[:, :, :T].transpose(1, 2).contiguous()   # 去掉补的部分，变回 [B, T, 头, dv]
    return out.to(dtype), S
