import numpy as np


def sequence_logprob(logits, tokens, mask):
    lg = logits[:, :-1].astype(np.float64)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    picked = np.take_along_axis(logp, tokens[:, 1:, None], axis=-1)[..., 0]
    return (picked * mask[:, 1:]).sum(axis=1)


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    loss = np.logaddexp(0.0, -beta * margin).mean()
    return float(loss), float((margin > 0).mean())


def grpo_advantages(rewards, group_size, eps=1e-6):
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    return ((r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + eps)).ravel()
