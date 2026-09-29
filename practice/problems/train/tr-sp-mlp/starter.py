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
    pass


def sp_backward(dout_shards, cache, comm):
    pass
