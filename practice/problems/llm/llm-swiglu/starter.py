import numpy as np


def silu(z):
    return z / (1 + np.exp(-z))


def swiglu_ffn(x, w_gate, w_up, w_down):
    pass


def swiglu_ffn_merged(x, w_gate_up, w_down):
    pass


def ffn_dim_for_params(d, multiple_of=256):
    pass
