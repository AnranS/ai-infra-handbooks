import numpy as np


def mha(x, wq, wk, wv, wo, n_heads, causal=True, mask=None):
    B, T, d = x.shape
    dh = d // n_heads
    q, k, v = x @ wq, x @ wk, x @ wv
    # TODO：拆成多头、因果掩码和 padding 掩码、稳定的 softmax、拼回并乘 wo
    pass
