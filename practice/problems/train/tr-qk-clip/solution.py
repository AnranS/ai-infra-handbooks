import math


def _matmul(A, B):
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def max_logit(x, wq, wk):
    q, k = _matmul(x, wq), _matmul(x, wk)
    d = len(wq[0])
    return max(sum(a * b for a, b in zip(qi, kj)) / math.sqrt(d) for qi in q for kj in k)


def qk_clip(x, Wq, Wk, tau, alpha=0.5):
    new_q, new_k, gammas = [], [], []
    for wq, wk in zip(Wq, Wk):
        gamma = min(1.0, tau / max_logit(x, wq, wk))
        gammas.append(gamma)
        new_q.append([[v * gamma ** alpha for v in row] for row in wq])
        new_k.append([[v * gamma ** (1 - alpha) for v in row] for row in wk])
    return new_q, new_k, gammas
