from __future__ import annotations

from typing import Tuple

import torch
from minisgl.kernel import fused_add_rmsnorm, rmsnorm

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """把"残差相加"和"归一化"融合成一个算子，同时返回归一化结果和新的残差。"""

    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:  # 第一层：还没有残差，x 本身就是残差
            return rmsnorm(x, self.weight, self.eps), x
        fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
