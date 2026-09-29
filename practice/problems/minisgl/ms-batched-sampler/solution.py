import numpy as np


def sample_batch(logits, temperature, top_k, top_p, u):
    logits = np.asarray(logits, dtype=np.float64)
    B, V = logits.shape
    t = np.asarray(temperature, dtype=np.float64)
    k = np.asarray(top_k)
    p = np.asarray(top_p, dtype=np.float64)
    greedy = (t <= 0) | ((k == 1) & (p == 1.0))
    out = np.argmax(logits, axis=1).astype(np.int64)
    rows = np.nonzero(~greedy)[0]
    if len(rows) == 0:
        return out
    z = logits[rows] / t[rows, None]
    probs = np.exp(z - z.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    cols = np.broadcast_to(np.arange(V), probs.shape)
    order = np.lexsort((cols, -probs), axis=-1)                   # 每行：概率降序，并列按下标
    sorted_p = np.take_along_axis(probs, order, axis=1)
    kk = np.where(k[rows] > 0, k[rows], V)
    keep_sorted = np.arange(V)[None, :] < kk[:, None]
    sorted_p = np.where(keep_sorted, sorted_p, 0.0)
    before = np.cumsum(sorted_p, axis=1) - sorted_p               # 加上自己之前的累计
    pp = p[rows][:, None]
    keep_sorted &= (pp >= 1.0) | (before < pp - 1e-12)
    keep = np.zeros_like(keep_sorted)
    np.put_along_axis(keep, order, keep_sorted, axis=1)
    probs = np.where(keep, probs, 0.0)
    probs /= probs.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)
    idx = np.argmax(cdf > np.asarray(u)[rows, None], axis=1)
    last = V - 1 - np.argmax(keep[:, ::-1], axis=1)               # 浮点误差时兜底：最后一个保留的 token
    idx = np.where(cdf[:, -1] > np.asarray(u)[rows], idx, last)
    out[rows] = idx
    return out
