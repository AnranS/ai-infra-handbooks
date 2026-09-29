import numpy as np


def mha(x, wq, wk, wv, wo, n_heads, causal=True, mask=None):
    B, T, d = x.shape
    H, dh = n_heads, d // n_heads

    def split(t):
        return t.reshape(B, T, H, dh).transpose(0, 2, 1, 3)

    q, k, v = split(x @ wq), split(x @ wk), split(x @ wv)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)             # (B, H, T, T)
    allowed = np.ones((B, 1, T, T), dtype=bool)
    if causal:
        allowed &= np.tril(np.ones((T, T), dtype=bool))[None, None]
    if mask is not None:
        allowed &= np.asarray(mask, dtype=bool)[:, None, None, :]
    scores = np.where(allowed, scores, -np.inf)
    m = scores.max(axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.where(allowed, np.exp(scores - m), 0.0)
    s = e.sum(axis=-1, keepdims=True)
    p = np.divide(e, s, out=np.zeros_like(e), where=s > 0)
    out = (p @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
    return out @ wo
