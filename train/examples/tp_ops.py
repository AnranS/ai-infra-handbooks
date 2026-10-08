import torch
import torch.distributed as dist


class CopyToTP(torch.autograd.Function):
    """Megatron 的 f：前向是恒等（每个 rank 拿同一份输入），反向把各 rank 对输入的梯度 all-reduce 求和。"""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        dist.all_reduce(grad)
        return grad


class ReduceFromTP(torch.autograd.Function):
    """Megatron 的 g：前向 all-reduce 求和（合并行切分矩阵乘的部分和），反向是恒等。"""

    @staticmethod
    def forward(ctx, x):
        x = x.clone()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


class GatherSeq(torch.autograd.Function):
    """序列并行的 g-bar：前向沿序列维 all-gather（拼回完整序列），反向 reduce-scatter。"""

    @staticmethod
    def forward(ctx, x):
        out = x.new_empty(x.shape[0] * dist.get_world_size(), *x.shape[1:])
        dist.all_gather_into_tensor(out, x.contiguous())
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty(grad.shape[0] // dist.get_world_size(), *grad.shape[1:])
        dist.reduce_scatter_tensor(out, grad.contiguous())
        return out


class ScatterSeq(torch.autograd.Function):
    """序列并行的 f-bar：前向 reduce-scatter（求和并沿序列维切开），反向 all-gather。"""

    @staticmethod
    def forward(ctx, x):
        out = x.new_empty(x.shape[0] // dist.get_world_size(), *x.shape[1:])
        dist.reduce_scatter_tensor(out, x.contiguous())
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty(grad.shape[0] * dist.get_world_size(), *grad.shape[1:])
        dist.all_gather_into_tensor(out, grad.contiguous())
        return out
