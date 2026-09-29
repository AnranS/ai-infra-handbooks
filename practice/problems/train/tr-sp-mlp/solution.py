import numpy as np


def act(x):
    """GELU 的 sigmoid 近似：x · σ(1.702x)"""
    return x / (1 + np.exp(-1.702 * x))


def act_grad(x):
    s = 1 / (1 + np.exp(-1.702 * x))
    return s + 1.702 * x * s * (1 - s)


class Comm:
    """t 个 rank 之间的集合通信（模拟）：输入是每个 rank 的数组组成的列表，输出也是"""

    def __init__(self, t):
        self.t = t
        self.log = []

    def all_gather(self, shards):
        full = np.concatenate(shards, axis=0)
        self.log.append(("all_gather", full.size))
        return [full.copy() for _ in range(self.t)]

    def reduce_scatter(self, partials):
        total = np.sum(partials, axis=0)
        self.log.append(("reduce_scatter", total.size))
        return [p.copy() for p in np.split(total, self.t, axis=0)]

    def all_reduce(self, partials):
        total = np.sum(partials, axis=0)
        self.log.append(("all_reduce", total.size))
        return [total.copy() for _ in range(self.t)]


def sp_forward(x_shards, g, W1_shards, W2_shards, comm, eps=1e-6):
    t = comm.t
    rms = [np.sqrt((x * x).mean(-1, keepdims=True) + eps) for x in x_shards]
    xhat = [x / r for x, r in zip(x_shards, rms)]
    Y = comm.all_gather([xh * g for xh in xhat])               # 进入 TP 区域：沿序列拼成完整的 (s, h)
    H = [Y[r] @ W1_shards[r].T for r in range(t)]              # 列切分：每个 rank 算 f/t 个中间特征
    A = [act(h) for h in H]
    Z = comm.reduce_scatter([A[r] @ W2_shards[r].T for r in range(t)])   # 行切分得到部分和：相加并切回序列分片
    out = [x + z for x, z in zip(x_shards, Z)]
    cache = dict(x=x_shards, rms=rms, xhat=xhat, g=g, Y=Y, H=H, A=A, W1=W1_shards, W2=W2_shards)
    return out, cache


def sp_backward(dout_shards, cache, comm):
    t = comm.t
    dZ = comm.all_gather(dout_shards)                          # reduce_scatter 的反向
    dW2 = [dZ[r].T @ cache["A"][r] for r in range(t)]
    dH = [(dZ[r] @ cache["W2"][r]) * act_grad(cache["H"][r]) for r in range(t)]
    dW1 = [dH[r].T @ cache["Y"][r] for r in range(t)]
    dY = comm.reduce_scatter([dH[r] @ cache["W1"][r] for r in range(t)])   # all_gather 的反向：部分和相加再切开
    g = cache["g"]
    dx, dg_part = [], []
    for r in range(t):                                         # RMSNorm 的反向：每行独立，在序列分片上就能算
        xh, dxh = cache["xhat"][r], dY[r] * g
        dg_part.append((dY[r] * xh).sum(0))
        dnorm = (dxh - xh * (dxh * xh).mean(-1, keepdims=True)) / cache["rms"][r]
        dx.append(dout_shards[r] + dnorm)
    dg = comm.all_reduce(dg_part)[0]                           # 每个 rank 只有自己那段序列的贡献
    return dx, dg, dW1, dW2
