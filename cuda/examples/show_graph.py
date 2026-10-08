import torch


def show_graph(gm, example_inputs):
    print(gm.code.strip())
    return gm.forward          # 返回一个可调用对象：这里直接用未优化的图


def mlp(x, w):
    return torch.nn.functional.silu(x @ w)


torch.compile(mlp, backend=show_graph)(torch.randn(4, 16), torch.randn(16, 16))
