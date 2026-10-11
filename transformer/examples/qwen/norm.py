"""RMSNorm：把每个向量缩放到"均方根为 1"，再逐维乘一个可学的比例。

Qwen3.5 存的是零中心的权重：文件里的 weight 是 w - 1，用的时候乘 (1 + weight)。
这样初始化为 0 就等于"不缩放"，bf16 下也比存 1.0x 的数精度高（1 附近的 bf16 间隔是 2^-7，0 附近细得多）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))                # 零中心：0 表示比例为 1
        self.eps = eps

    def forward(self, x):
        xf = x.float()                                              # 统计量一律用 fp32 算
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """线性注意力的输出用这个：先归一化，再乘 silu(门)。它的权重不是零中心的（初始为 1）。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * y * F.silu(gate.float())).type_as(x)


def l2norm(x, eps=1e-6):
    """把最后一维缩放到单位长度（线性注意力里 q、k 进记忆表之前用）"""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)
