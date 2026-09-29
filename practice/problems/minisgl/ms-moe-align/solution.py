import numpy as np


def moe_align_block_size(topk_ids, block_m, num_experts):
    flat = np.asarray(topk_ids).ravel().astype(np.int64)
    num_pairs = flat.size
    counts = np.bincount(flat, minlength=num_experts)
    padded = (counts + block_m - 1) // block_m * block_m
    seg_start = np.cumsum(padded) - padded
    order = np.argsort(flat, kind="stable")
    sorted_expert = flat[order]
    rank = np.arange(num_pairs) - (np.cumsum(counts) - counts)[sorted_expert]
    sorted_ids = np.full(int(padded.sum()), num_pairs, dtype=np.int32)
    sorted_ids[seg_start[sorted_expert] + rank] = order
    expert_ids = np.repeat(np.arange(num_experts), padded // block_m).astype(np.int32)
    return sorted_ids, expert_ids


def fused_moe_grouped(x, w, topk_ids, topk_weights, block_m):
    T, k = topk_ids.shape
    num_pairs = T * k
    sorted_ids, expert_ids = moe_align_block_size(topk_ids, block_m, w.shape[0])
    weights = np.asarray(topk_weights).ravel()
    out = np.zeros((T, w.shape[1]))
    for b, e in enumerate(expert_ids):
        ids = sorted_ids[b * block_m:(b + 1) * block_m]
        ids = ids[ids < num_pairs]
        if len(ids) == 0:
            continue
        tokens = ids // k
        y = x[tokens] @ w[e].T                      # 每个块一次矩阵乘
        np.add.at(out, tokens, weights[ids][:, None] * y)
    return out
