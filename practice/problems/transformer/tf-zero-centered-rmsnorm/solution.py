import torch
import torch.nn.functional as F


def rmsnorm(x, weight, eps=1e-6):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1 + weight.float())).type_as(x)


def gated_rmsnorm(x, gate, weight, eps=1e-6):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * y * F.silu(gate.float())).type_as(x)
