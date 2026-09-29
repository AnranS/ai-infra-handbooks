import numpy as np


def sample(dist, u):
    c = np.cumsum(np.asarray(dist, dtype=np.float64) / np.sum(dist))
    i = int(np.searchsorted(c, u, side="right"))
    return i if i < len(c) else int(np.nonzero(dist)[0][-1])


def verify(draft, q, p, u, u_resample, u_bonus):
    out = []
    for i, t in enumerate(draft):
        if u[i] < min(1.0, p[i][t] / q[i][t]):
            out.append(int(t))
            continue
        r = np.maximum(np.asarray(p[i], dtype=np.float64) - q[i], 0.0)
        out.append(sample(r, u_resample))
        return out
    out.append(sample(p[len(draft)], u_bonus))
    return out


def verify_greedy(draft, target_argmax):
    out = []
    for i, t in enumerate(draft):
        if t != target_argmax[i]:
            out.append(int(target_argmax[i]))
            return out
        out.append(int(t))
    out.append(int(target_argmax[len(draft)]))
    return out
