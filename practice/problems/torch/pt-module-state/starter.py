import torch
import torch.nn as nn


class Stack(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        # TODO：n_layers 个 nn.Linear(dim, dim)，以及名为 scale 的 buffer

    def forward(self, x):
        pass


def state_keys(model):
    pass


def load_partial(model, state):
    pass
