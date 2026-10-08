"""sampling.py —— 常用采样策略的实现。logits: [B, V]。"""

import torch


def apply_repetition_penalty(logits, prev_ids, penalty):
    """transformers 的定义：出现过的 token，正 logit 除以 penalty，负 logit 乘以 penalty。"""
    if penalty == 1.0:
        return logits
    score = logits.gather(1, prev_ids)
    score = torch.where(score > 0, score / penalty, score * penalty)
    return logits.scatter(1, prev_ids, score)


def top_k_filter(logits, k):
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1:]            # 第 k 大的值
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    cum_before = probs.cumsum(dim=-1) - probs                 # 排在它前面的 token 的累积概率
    remove_sorted = cum_before >= p                           # 前面已经够 p 了，它就不需要了
    remove = remove_sorted.scatter(1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def min_p_filter(logits, min_p):
    if min_p <= 0.0:
        return logits
    probs = logits.softmax(dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample_next(logits, prev_ids=None, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
                repetition_penalty=1.0, generator=None):
    """按 transformers 的顺序：重复惩罚 -> 温度 -> top-k -> top-p -> min-p -> 采样。"""
    if prev_ids is not None:
        logits = apply_repetition_penalty(logits, prev_ids, repetition_penalty)
    if temperature == 0.0:
        return logits.argmax(dim=-1)                          # 温度为 0 视为贪心
    logits = logits / temperature
    logits = min_p_filter(top_p_filter(top_k_filter(logits, top_k), top_p), min_p)
    return torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).squeeze(-1)
