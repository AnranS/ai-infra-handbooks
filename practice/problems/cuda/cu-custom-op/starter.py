import torch


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    residual.add_(x)
    r = residual.float()
    x.copy_((r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight)


# TODO：把上面两个函数注册成 torch.ops.practice.* 下的自定义算子，并补上 fake 实现和反向
