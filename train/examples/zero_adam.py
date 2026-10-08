import math

import torch
import torch.distributed as dist


class ZeroAdam:
    """ZeRO（第 2 级）的 Adam：梯度用 reduce-scatter 求和并切分，每个 rank 只保存、只更新自己那一片参数的
    主参数和 Adam 状态，更新完用 all-gather 把参数拼回来。"""

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.t = lr, betas, eps, 0
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.numel = sum(p.numel() for p in self.params)
        self.shard = math.ceil(self.numel / self.world)          # 每个 rank 负责的元素个数（末尾补零对齐）
        flat = self._flatten([p.data for p in self.params])
        lo = self.rank * self.shard
        self.master = flat[lo:lo + self.shard].clone()           # 只保存自己那一片（真实训练里这是 fp32 主参数）
        self.m = torch.zeros_like(self.master)
        self.v = torch.zeros_like(self.master)

    def _flatten(self, tensors):
        flat = torch.zeros(self.shard * self.world)
        flat[:self.numel] = torch.cat([t.flatten() for t in tensors])
        return flat

    def step(self):
        grads = self._flatten([p.grad for p in self.params])
        g = torch.empty(self.shard)
        dist.reduce_scatter_tensor(g, grads)                     # 求和并切分：每个 rank 只拿到自己那片的梯度
        g /= self.world                                          # 求平均
        self.t += 1
        b1, b2 = self.betas
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        m_hat = self.m / (1 - b1 ** self.t)
        v_hat = self.v / (1 - b2 ** self.t)
        self.master -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
        full = torch.empty(self.shard * self.world)
        dist.all_gather_into_tensor(full, self.master)           # 把各 rank 更新好的片拼回完整参数
        offset = 0
        for p in self.params:
            p.data.copy_(full[offset:offset + p.numel()].view_as(p))
            offset += p.numel()

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.master, self.m, self.v))
