import torch


def pad_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device) < lengths[:, None]


def mask_scores(scores, lengths):
    mask = pad_mask(lengths, scores.shape[-1])
    return scores.masked_fill(~mask[:, None, :], float("-inf"))


def last_token(h, lengths):
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.shape[-1])
    return h.gather(1, idx).squeeze(1)


def top_k_filter(logits, k):
    vals, idx = logits.topk(k, dim=-1)
    out = torch.full_like(logits, float("-inf"))
    return out.scatter_(-1, idx, vals)
