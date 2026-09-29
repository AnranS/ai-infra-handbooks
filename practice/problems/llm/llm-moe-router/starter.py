import numpy as np


def swiglu(x, w_gate, w_up, w_down):
    g = x @ w_gate
    return (g / (1 + np.exp(-g)) * (x @ w_up)) @ w_down


def moe_forward(x, router_w, experts, top_k, norm_topk_prob=True):
    pass
