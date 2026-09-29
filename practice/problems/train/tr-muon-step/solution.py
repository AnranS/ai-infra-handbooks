import math


def _matmul(A, B):
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def _transpose(A):
    return [list(r) for r in zip(*A)]


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = math.sqrt(sum(v * v for row in G for v in row)) + 1e-7
    X = [[v / norm for v in row] for row in G]
    transposed = len(X) > len(X[0])
    if transposed:
        X = _transpose(X)
    for _ in range(steps):
        A = _matmul(X, _transpose(X))
        AA = _matmul(A, A)
        B = [[b * x + c * y for x, y in zip(ra, rb)] for ra, rb in zip(A, AA)]
        BX = _matmul(B, X)
        X = [[a * x + y for x, y in zip(rx, rb)] for rx, rb in zip(X, BX)]
    return _transpose(X) if transposed else X


def muon_step(W, grad, buf, lr, momentum=0.95, weight_decay=0.0):
    buf = [[momentum * m + g for m, g in zip(rm, rg)] for rm, rg in zip(buf, grad)]
    U = newton_schulz([[g + momentum * m for g, m in zip(rg, rm)] for rg, rm in zip(grad, buf)])
    scale = lr * 0.2 * math.sqrt(max(len(W), len(W[0])))
    W = [[w * (1 - lr * weight_decay) - scale * u for w, u in zip(rw, ru)] for rw, ru in zip(W, U)]
    return W, buf
