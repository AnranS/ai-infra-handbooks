import torch

compiles = []


def counting_backend(gm, example_inputs):
    compiles.append(len(example_inputs))
    return gm.forward


def mlp(x, w):
    return torch.nn.functional.silu(x @ w)


f = torch.compile(mlp, backend=counting_backend)
w = torch.randn(16, 16)
for batch in (4, 4, 8, 16, 32):
    f(torch.randn(batch, 16), w)
    print(f"batch={batch:<3} 累计编译 {len(compiles)} 次")
