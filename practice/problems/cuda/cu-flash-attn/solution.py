import numpy as np


def flash_attn_fwd(q, k, v, causal=False, Br=64, Bc=64):
    T, d = q.shape
    S = k.shape[0]
    scale = 1.0 / np.sqrt(d)
    o = np.empty((T, v.shape[1]), dtype=np.float64)
    lse = np.empty(T, dtype=np.float64)
    for i0 in range(0, T, Br):
        qi = q[i0:i0 + Br]
        rows = len(qi)
        qpos = S - T + i0 + np.arange(rows)
        m = np.full(rows, -np.inf)
        l = np.zeros(rows)
        acc = np.zeros((rows, v.shape[1]))
        for j0 in range(0, S, Bc):
            if causal and j0 > qpos[-1]:
                break
            s = (qi @ k[j0:j0 + Bc].T) * scale
            if causal:
                kpos = j0 + np.arange(s.shape[1])
                s = np.where(kpos[None, :] <= qpos[:, None], s, -np.inf)
            m_new = np.maximum(m, s.max(axis=1))
            p = np.exp(s - m_new[:, None])
            alpha = np.exp(m - m_new)
            l = l * alpha + p.sum(axis=1)
            acc = acc * alpha[:, None] + p @ v[j0:j0 + Bc]
            m = m_new
        o[i0:i0 + rows] = acc / l[:, None]
        lse[i0:i0 + rows] = m + np.log(l)
    return o, lse
