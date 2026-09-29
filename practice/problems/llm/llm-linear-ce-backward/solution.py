import numpy as np


def forward_backward(H, W, b, y):
    N = H.shape[0]
    Z = H @ W + b
    Z = Z - Z.max(axis=1, keepdims=True)              # 减最大值，数值稳定
    lse = np.log(np.exp(Z).sum(axis=1, keepdims=True))
    rows = np.arange(N)
    loss = float(-(Z[rows, y] - lse[:, 0]).mean())     # 直接用 log-softmax，避免 log(0)
    dZ = np.exp(Z - lse)                              # softmax
    dZ[rows, y] -= 1.0
    dZ /= N
    return loss, dZ @ W.T, H.T @ dZ, dZ.sum(axis=0)
