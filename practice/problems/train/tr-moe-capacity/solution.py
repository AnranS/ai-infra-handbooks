import math

import numpy as np


def capacity(T, k, E, factor):
    return math.ceil(factor * T * k / E)


def dispatch(topk_idx, topk_w, E, cap, policy="order"):
    T, k = topk_idx.shape
    slot = np.full((T, k), -1)
    for e in range(E):
        ts, js = np.nonzero(topk_idx == e)            # 选中专家 e 的所有份，已经按 (t, j) 排好
        if policy == "score":
            order = sorted(range(len(ts)), key=lambda i: (-topk_w[ts[i], js[i]], ts[i], js[i]))
        else:
            order = range(len(ts))
        for rank, i in enumerate(order):
            if rank < cap:                            # 前 cap 份进缓冲区，其余丢弃
                slot[ts[i], js[i]] = rank
    return slot


def combine(expert_out, topk_idx, topk_w, slot):
    T, k = topk_idx.shape
    out = np.zeros((T, expert_out.shape[2]))
    for t in range(T):
        for j in range(k):
            if slot[t, j] >= 0:
                out[t] += topk_w[t, j] * expert_out[topk_idx[t, j], slot[t, j]]
    return out


def aux_loss(logits, topk_idx):
    T, E = logits.shape
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    f = np.bincount(topk_idx.ravel(), minlength=E) / topk_idx.size
    P = p.mean(0)
    loss = E * float(f @ P)
    a = E * f                                         # dL/dP_e
    dlogits = p * (a - (p * a).sum(1, keepdims=True)) / T     # softmax 的雅可比，再对 token 取平均
    return loss, dlogits
