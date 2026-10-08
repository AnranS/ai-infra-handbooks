import torch


def step(x, w):
    h = x @ w
    if h.sum().item() > 0:
        h = h * 2
    return torch.relu(h)


def step_nobreak(x, w):
    h = x @ w
    h = torch.where(h.sum() > 0, h * 2, h)   # 条件是一个 0 维张量，留在图里
    return torch.relu(h)


torch.manual_seed(0)
compiled = torch.compile(step_nobreak, fullgraph=True, backend="eager")
ok = all(torch.allclose(compiled(x, w), step(x, w)) for x, w in [(torch.randn(4, 16), torch.randn(16, 16)) for _ in range(5)])
print("fullgraph 编译成功，结果一致：", ok)
