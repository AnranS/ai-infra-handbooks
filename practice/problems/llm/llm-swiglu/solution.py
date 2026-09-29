import numpy as np


def silu(z):
    return z * (0.5 * (1.0 + np.tanh(0.5 * z)))


def swiglu_ffn(x, w_gate, w_up, w_down):
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


def swiglu_ffn_merged(x, w_gate_up, w_down):
    d_ff = w_gate_up.shape[1] // 2
    h = x @ w_gate_up
    return (silu(h[:, :d_ff]) * h[:, d_ff:]) @ w_down


def ffn_dim_for_params(d, multiple_of=256):
    hidden = int(2 * 4 * d / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)
