import torch
from torch.utils._python_dispatch import TorchDispatchMode


class CountOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        self.counts[name] = self.counts.get(name, 0) + 1
        return func(*args, **(kwargs or {}))


q = k = v = torch.randn(1, 8, 128, 64)
with CountOps() as c:
    torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
for name, n in sorted(c.counts.items()):
    print(name, n)
