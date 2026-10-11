import torch


def filter_logits(logits, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, presence_penalty=0.0, generated=()):
    logits = logits.float().clone()
    if generated:
        seen = torch.tensor(sorted(set(generated)))
        s = logits[seen]
        s = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)
        logits[seen] = s - presence_penalty
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(top_k).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p < 1.0:
        sorted_logits, order = logits.sort(descending=True)
        probs = sorted_logits.softmax(-1)
        drop = probs.cumsum(-1) - probs > top_p
        sorted_logits[drop] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
    return logits
