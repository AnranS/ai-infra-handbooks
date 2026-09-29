import math

from checker import check, check_close
from solution import muon_step, newton_schulz


def close_matrix(actual, expected, what, atol=1e-6):
    check(len(actual), len(expected), f"{what}：行数")
    for i, (ra, re) in enumerate(zip(actual, expected)):
        check(len(ra), len(re), f"{what}：第 {i} 行的列数")
        for j, (a, e) in enumerate(zip(ra, re)):
            check_close(a, e, rtol=1e-6, atol=atol, what=f"{what}[{i}][{j}]")


def gram(X):
    return [[sum(a * b for a, b in zip(r1, r2)) for r2 in X] for r1 in X]


def test_example():
    X = newton_schulz([[3.0, 0.0], [0.0, 0.1]])
    close_matrix(X, [[0.6970, 0.0], [0.0, 1.1288]], "对角矩阵", atol=2e-4)


def test_orthogonal_ish():
    G = [[1.0, 2.0, 0.5, -1.0], [0.3, -0.2, 1.5, 0.7], [2.0, 0.1, -0.4, 0.2]]
    X = newton_schulz(G)
    M = gram(X)                                              # X Xᵀ 的对角线是奇异值的平方
    for i in range(3):
        if not 0.45 <= M[i][i] <= 1.5:
            raise AssertionError(f"X Xᵀ 的对角线应在 1 附近，第 {i} 个是 {M[i][i]:.3f}")
    check_close(newton_schulz(G, steps=0)[0][0], 1.0 / math.sqrt(sum(v * v for r in G for v in r)), rtol=1e-6,
                what="0 步时只做了除以 Frobenius 范数")


def test_tall_matrix():
    G = [[1.0, 0.5], [0.2, -1.0], [0.3, 0.3], [2.0, 0.0]]     # 4 × 2：行数多于列数
    X = newton_schulz(G)
    check(len(X), 4, "形状保持 4 × 2（行数）")
    check(len(X[0]), 2, "形状保持 4 × 2（列数）")
    Xt = newton_schulz([list(r) for r in zip(*G)])           # 对转置做正交化再转置回来，结果相同
    close_matrix(X, [list(r) for r in zip(*Xt)], "行多于列时先转置")


def test_muon_step():
    W = [[0.5, -0.5], [1.0, 0.0]]
    g = [[0.1, 0.2], [-0.3, 0.05]]
    buf0 = [[0.0, 0.0], [0.0, 0.0]]
    W1, buf1 = muon_step(W, g, buf0, lr=0.1, momentum=0.9, weight_decay=0.1)
    close_matrix(buf1, g, "第一步的动量就是梯度")
    U = newton_schulz([[1.9 * v for v in r] for r in g])     # g + 0.9·buf，buf = g
    s = 0.1 * 0.2 * math.sqrt(2)
    close_matrix(W1, [[w * 0.99 - s * u for w, u in zip(rw, ru)] for rw, ru in zip(W, U)], "第一步的权重")
    W2, buf2 = muon_step(W1, g, buf1, lr=0.1, momentum=0.9)
    close_matrix(buf2, [[1.9 * v for v in r] for r in g], "第二步的动量 0.9·g + g")
