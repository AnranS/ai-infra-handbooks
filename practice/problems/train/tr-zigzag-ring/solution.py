import numpy as np


def zigzag_order(P):
    return [[r, 2 * P - 1 - r] for r in range(P)]


def _attn(qc, kc, vc, causal):
    """一个 query 块对一个 KV 块的注意力：返回 (输出, 每行的 lse)"""
    s = qc @ kc.T / np.sqrt(qc.shape[1])
    if causal:
        s = np.where(np.tril(np.ones(s.shape, dtype=bool)), s, -np.inf)
    m = s.max(1, keepdims=True)
    p = np.exp(s - m)
    return p @ vc / p.sum(1, keepdims=True), (m + np.log(p.sum(1, keepdims=True)))[:, 0]


def ring_attention_zigzag(q, k, v, P):
    S = q.shape[0]
    c = S // (2 * P)
    blk = lambda x, i: x[i * c:(i + 1) * c]
    own = zigzag_order(P)
    kv = [[(j, blk(k, j), blk(v, j)) for j in own[r]] for r in range(P)]     # 每个 rank 当前手里的 KV 块
    acc = [[None, None] for _ in range(P)]                                   # 每个 query 块的 (输出, lse)
    work = []
    for step in range(P):
        row = []
        for r in range(P):
            w = 0
            for qi, a in enumerate(own[r]):
                for b, kc, vc in kv[r]:
                    if b > a:                                                # KV 在 query 之后：整块跳过
                        continue
                    o, l = _attn(blk(q, a), kc, vc, causal=(a == b))
                    w += c * c if b < a else c * (c + 1) // 2
                    if acc[r][qi] is None:
                        acc[r][qi] = (o, l)
                    else:                                                    # log-sum-exp 合并
                        o1, l1 = acc[r][qi]
                        new = np.logaddexp(l1, l)
                        acc[r][qi] = (o1 * np.exp(l1 - new)[:, None] + o * np.exp(l - new)[:, None], new)
            row.append(w)
        work.append(row)
        kv = [kv[(r - 1) % P] for r in range(P)]                            # 传给下一个 rank：rank r 收到 r-1 的
    out = np.zeros((S, v.shape[1]))
    for r in range(P):
        for qi, a in enumerate(own[r]):
            out[a * c:(a + 1) * c] = acc[r][qi][0]
    return out, work
