import torch

records = []


def pack(t):
    records.append((tuple(t.shape), t.numel() * t.element_size()))
    return t


torch.manual_seed(0)
lin = torch.nn.Linear(4096, 4096)
x = torch.randn(8, 4096, requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    y = torch.nn.functional.gelu(lin(x))
for shape, n in records:
    print(shape, n)
