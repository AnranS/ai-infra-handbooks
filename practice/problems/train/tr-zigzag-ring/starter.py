import numpy as np


def zigzag_order(P):
    return [[2 * r, 2 * r + 1] for r in range(P)]         # 按顺序切：负载不均


def ring_attention_zigzag(q, k, v, P):
    pass
