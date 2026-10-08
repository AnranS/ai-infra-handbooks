import torch
import torch.nn as nn


class Stack(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()                      # 必须是第一句，否则下面的赋值会报错
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.register_buffer("scale", torch.tensor(dim**-0.5))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).relu()
        return x * self.scale


def state_keys(model):
    return list(model.state_dict().keys())


def load_partial(model, state):
    out = model.load_state_dict(state, strict=False)
    return sorted(out.missing_keys), sorted(out.unexpected_keys)
