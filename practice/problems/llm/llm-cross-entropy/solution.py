import numpy as np


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def cross_entropy(logits, targets, ignore_index=-100):
    targets = np.asarray(targets)
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    logp = _log_softmax(np.asarray(logits)[mask])
    return float(-logp[np.arange(mask.sum()), targets[mask]].mean())


def kl_divergence(p_logits, q_logits):
    log_p, log_q = _log_softmax(p_logits), _log_softmax(q_logits)
    return (np.exp(log_p) * (log_p - log_q)).sum(axis=-1)


def perplexity(logits, targets, ignore_index=-100):
    return float(np.exp(cross_entropy(logits, targets, ignore_index)))
