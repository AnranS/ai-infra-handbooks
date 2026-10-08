import torch
from torch.utils.checkpoint import checkpoint


def mlp(x, w1, w2):
    return torch.nn.functional.silu(x @ w1) @ w2


saved = []


def pack(t):
    saved.append(t.numel() * t.element_size())
    return t


def unpack(t):
    return t


torch.manual_seed(0)
X = torch.randn(64, 256, requires_grad=True)
W1 = torch.randn(256, 1024, requires_grad=True)
W2 = torch.randn(1024, 256, requires_grad=True)

with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    out = mlp(X, W1, W2)
print(f"普通前向：保存了 {len(saved)} 个张量，共 {sum(saved)} 字节")

saved.clear()
with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    out2 = checkpoint(mlp, X, W1, W2, use_reentrant=False)
print(f"激活重计算：保存了 {len(saved)} 个张量，共 {sum(saved)} 字节")
out2.sum().backward()
print("反向照常完成，X.grad 形状", tuple(X.grad.shape))
