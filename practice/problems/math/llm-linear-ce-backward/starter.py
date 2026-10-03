import numpy as np


def forward_backward(H, W, b, y):
    Z = H @ W + b
    # TODO：数值稳定的 softmax、交叉熵，以及 dH、dW、db
    pass
