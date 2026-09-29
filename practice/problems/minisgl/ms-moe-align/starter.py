import numpy as np


def moe_align_block_size(topk_ids, block_m, num_experts):
    pass


def fused_moe_grouped(x, w, topk_ids, topk_weights, block_m):
    # 逐个 pair 计算：结果对，但没有按块分组
    T, k = topk_ids.shape
    out = np.zeros((T, w.shape[1]))
    for i in range(T * k):
        t, e = i // k, topk_ids.ravel()[i]
        out[t] += topk_weights.ravel()[i] * (w[e] @ x[t])
    return out
