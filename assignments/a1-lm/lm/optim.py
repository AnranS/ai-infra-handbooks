"""优化器与训练工具。不允许调用 torch.optim.AdamW / torch.nn.utils.clip_grad_norm_ 等现成实现。"""

import math  # noqa: F401

import torch


class AdamW(torch.optim.Optimizer):
    """与 torch.optim.AdamW 数值一致（测试逐元素比较）：先做解耦的权重衰减 p ← p·(1 − lr·wd)，
    再做带偏置修正的 Adam 更新。参数：lr、betas=(0.9, 0.999)、eps=1e-8、weight_decay=0.01。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError


def lr_schedule(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """step < warmup：从 0 线性升到 max_lr（step=0 时为 0）；warmup ≤ step ≤ total：余弦从 max_lr 降到 min_lr；
    step > total：保持 min_lr。"""
    raise NotImplementedError


def clip_grad_norm(params, max_norm: float) -> float:
    """所有参数的梯度拼在一起算 L2 范数；超过 max_norm 时整体乘以 max_norm / (范数 + 1e-6)。返回裁剪前的范数。"""
    raise NotImplementedError
