import torch

x = torch.randn(3)
print(torch._C._dispatch_keys(x))
