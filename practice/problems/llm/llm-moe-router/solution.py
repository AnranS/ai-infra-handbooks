import numpy as np


def swiglu(x, w_gate, w_up, w_down):
    g = x @ w_gate
    return (g / (1 + np.exp(-g)) * (x @ w_up)) @ w_down


def moe_forward(x, router_w, experts, top_k, norm_topk_prob=True):
    N, E = x.shape[0], router_w.shape[1]
    logits = x @ router_w
    logits = logits - logits.max(-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(-1, keepdims=True)
    topk_ids = np.argsort(-probs, axis=-1, kind="stable")[:, :top_k]
    topk_w = np.take_along_axis(probs, topk_ids, axis=-1)
    if norm_topk_prob:
        topk_w = topk_w / topk_w.sum(-1, keepdims=True)
    out = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    tokens_per_expert = []
    for e in range(E):
        rows, slots = np.nonzero(topk_ids == e)
        tokens_per_expert.append(sorted(rows.tolist()))
        if len(rows):
            out[rows] += topk_w[rows, slots, None] * swiglu(x[rows], *experts[e])
    f = np.bincount(topk_ids.ravel(), minlength=E) / (N * top_k)
    P = probs.mean(axis=0)
    return {"out": out, "topk_ids": topk_ids, "topk_weights": topk_w, "tokens_per_expert": tokens_per_expert,
            "aux_loss": float(E * np.sum(f * P))}
