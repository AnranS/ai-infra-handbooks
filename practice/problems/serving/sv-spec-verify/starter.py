import numpy as np


def sample(dist, u):
    c = np.cumsum(np.asarray(dist, dtype=np.float64) / np.sum(dist))
    i = int(np.searchsorted(c, u, side="right"))
    return i if i < len(c) else int(np.nonzero(dist)[0][-1])


def verify(draft, q, p, u, u_resample, u_bonus):
    out = []
    for i, t in enumerate(draft):          # 只接受目标模型也认为概率最大的 token：分布被改变了
        if int(np.argmax(p[i])) == t:
            out.append(t)
        else:
            out.append(int(np.argmax(p[i])))
            return out
    out.append(int(np.argmax(p[len(draft)])))
    return out


def verify_greedy(draft, target_argmax):
    pass
