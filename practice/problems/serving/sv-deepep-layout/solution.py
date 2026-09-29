import numpy as np


def dispatch_layout(topk_idx, num_experts, num_ranks):
    topk_idx = np.asarray(topk_idx)
    local = num_experts // num_ranks
    T = topk_idx.shape[0]
    valid = topk_idx >= 0
    in_rank = np.zeros((T, num_ranks), dtype=bool)
    rows = np.nonzero(valid)[0]
    in_rank[rows, topk_idx[valid] // local] = True           # 同一个 token 去同一张卡只记一次
    tokens_per_expert = np.bincount(topk_idx[valid], minlength=num_experts)
    return in_rank.sum(0), tokens_per_expert, in_rank


def recv_plan(all_topk, num_experts):
    R = len(all_topk)
    counts = np.array([dispatch_layout(t, num_experts, R)[0] for t in all_topk]).reshape(R, R)
    offsets = np.cumsum(counts, axis=0) - counts               # 每个目标卡的接收缓冲区里，按来源卡的编号依次排
    return counts, offsets, counts.sum(0)


def ll_slots(topk_idx, src_rank, num_experts, num_ranks, max_tokens):
    topk_idx = np.asarray(topk_idx)
    slots = np.full(topk_idx.shape, -1)
    used = np.zeros(num_experts, dtype=int)                  # 本卡发往每个专家的份数（src_rank 决定写哪一组槽位，槽位号只看份数）
    for t in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            e = topk_idx[t, j]
            if e < 0:
                continue
            if used[e] >= max_tokens:
                raise OverflowError(f"发往专家 {e} 的 token 超过了 {max_tokens} 个槽位")
            slots[t, j] = used[e]
            used[e] += 1
    return slots


def ll_buffer_bytes(num_experts, num_ranks, max_tokens, msg_bytes):
    return num_experts // num_ranks * num_ranks * max_tokens * msg_bytes
