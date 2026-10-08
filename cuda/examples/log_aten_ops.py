import torch
from torch.utils._python_dispatch import TorchDispatchMode


class LogOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


torch.manual_seed(0)
x = torch.randn(2, 16)
w = torch.ones(16)
lin = torch.nn.Linear(16, 32, bias=False)
with LogOps() as log:
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w   # RMSNorm
    y = torch.nn.functional.silu(lin(h))                               # Linear + SiLU
for op in log.ops:
    print(op)
