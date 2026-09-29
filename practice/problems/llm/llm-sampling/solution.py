import numpy as np


def sample(logits, u, temperature=1.0, top_k=-1, top_p=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    if temperature == 0:
        return int(np.argmax(logits))
    z = logits / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    n = len(p)
    order = np.lexsort((np.arange(n), -p))      # 概率降序，并列按下标升序
    if top_k > 0:
        p[order[top_k:]] = 0.0
    if top_p < 1.0:
        c = np.cumsum(p[order])
        keep = int(np.searchsorted(c, top_p - 1e-12)) + 1
        p[order[keep:]] = 0.0
    p /= p.sum()
    idx = int(np.searchsorted(np.cumsum(p), u, side="right"))   # 第一个累计和 > u 的位置
    return idx if idx < n else int(np.nonzero(p)[0][-1])        # 浮点误差让累计和略小于 1 时
