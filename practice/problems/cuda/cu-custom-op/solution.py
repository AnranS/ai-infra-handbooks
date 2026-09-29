import torch


@torch.library.custom_op("practice::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]


@silu_and_mul.register_fake
def _(x):
    return x.new_empty(*x.shape[:-1], x.shape[-1] // 2)       # 只描述输出的形状和类型


def _setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def _backward(ctx, grad):
    (x,) = ctx.saved_tensors
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    s = torch.sigmoid(gate)
    d_gate = grad * up * s * (1 + gate * (1 - s))              # silu'(a) = σ(a)(1 + a(1 - σ(a)))
    d_up = grad * gate * s
    return torch.cat([d_gate, d_up], dim=-1)


silu_and_mul.register_autograd(_backward, setup_context=_setup_context)


@torch.library.custom_op("practice::fused_add_rms_norm", mutates_args=("x", "residual"))
def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    residual.add_(x)
    r = residual.float()
    x.copy_((r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight)


@fused_add_rms_norm.register_fake
def _(x, residual, weight, eps):
    return None                                                # 没有输出，形状上什么都不用推导
