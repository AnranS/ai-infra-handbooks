"""Muon：动量 + Newton-Schulz 正交化，只用于隐藏层的二维权重矩阵"""
import math

import torch


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """近似 G 的极分解因子 U Vᵀ（G = U S Vᵀ）：把所有奇异值推到 1 附近，只用矩阵乘"""
    a, b, c = 3.4445, -4.7750, 2.0315                       # 五次多项式的系数：收敛快，奇异值落在 1 附近的一个区间
    X = G / (G.norm() + 1e-7)                               # 先缩放到谱范数 ≤ 1
    transposed = G.shape[0] > G.shape[1]
    if transposed:                                          # 让 X Xᵀ 是较小的那个方阵
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X                 # X ← a·X + b·(X Xᵀ)X + c·(X Xᵀ)²X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """每个矩阵：m ← μ·m + g；更新 = NS(g + μ·m)（Nesterov），再乘 0.2·√max(行, 列)，让更新的均方根和 AdamW 相当"""

    def __init__(self, params, lr, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                m = self.state[p].setdefault("momentum", torch.zeros_like(p))
                m.mul_(group["momentum"]).add_(p.grad)
                update = newton_schulz(p.grad + group["momentum"] * m)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * 0.2 * math.sqrt(max(p.shape)))
