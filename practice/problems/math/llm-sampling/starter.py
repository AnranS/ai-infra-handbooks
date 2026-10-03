import numpy as np


def sample(logits, u, temperature=1.0, top_k=-1, top_p=1.0):
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return int(np.searchsorted(np.cumsum(p), u, side="right"))
