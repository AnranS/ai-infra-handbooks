import numpy as np


def partial_attention(q, k, v, q_pos, k_pos, causal):
    T, d = q.shape
    s = q @ k.T / np.sqrt(d)
    if causal:
        s = np.where(np.asarray(k_pos)[None, :] <= np.asarray(q_pos)[:, None], s, -np.inf)
    if s.shape[1] == 0:
        return np.zeros((T, v.shape[1])), np.full(T, -np.inf)
    m = s.max(axis=1)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(s - m_safe[:, None])
    tot = e.sum(axis=1)
    o = np.divide(e @ v, tot[:, None], out=np.zeros((T, v.shape[1])), where=tot[:, None] > 0)
    with np.errstate(divide="ignore"):
        lse = np.where(tot > 0, m_safe + np.log(np.where(tot > 0, tot, 1.0)), -np.inf)
    return o, lse


def merge(parts):
    L = np.stack([p[1] for p in parts])                 # (P, T)
    O = np.stack([p[0] for p in parts])                 # (P, T, d)
    M = L.max(axis=0)
    M_safe = np.where(np.isfinite(M), M, 0.0)
    w = np.exp(L - M_safe)
    tot = w.sum(axis=0)
    o = np.divide((w[..., None] * O).sum(axis=0), tot[:, None], out=np.zeros(O.shape[1:]), where=tot[:, None] > 0)
    with np.errstate(divide="ignore"):
        lse = np.where(tot > 0, M_safe + np.log(np.where(tot > 0, tot, 1.0)), -np.inf)
    return o, lse


def context_parallel_attention(q, k, v, n, causal):
    S, T = len(k), len(q)
    q_pos = np.arange(S - T, S)
    parts = []
    for idx in np.array_split(np.arange(S), n):
        parts.append(partial_attention(q, k[idx], v[idx], q_pos, idx, causal))
    return merge(parts)
