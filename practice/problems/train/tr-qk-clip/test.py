import copy
import math

from checker import check, check_close
from solution import max_logit, qk_clip


def test_example():
    x = [[1.0, 0.0], [0.0, 1.0]]
    new_q, new_k, g = qk_clip(x, [[[20.0], [0.0]]], [[[10.0], [0.0]]], tau=100.0)
    check_close(g[0], 0.5, what="γ = 100 / 200")
    check_close(new_q[0][0][0], 20 * math.sqrt(0.5), what="W_q 乘 γ^0.5")
    check_close(new_k[0][0][0], 10 * math.sqrt(0.5), what="W_k 乘 γ^0.5")


def test_max_logit():
    x = [[1.0, 2.0], [-1.0, 0.5], [0.0, -3.0]]
    wq = [[1.0, 0.0], [0.0, 1.0]]
    wk = [[2.0, 1.0], [0.0, -1.0]]
    # q = x，k = [[2, -1], [-2, -1.5], [0, 3]]；q_i·k_j 的最大值是 (0,-3)·(-2,-1.5) = 4.5 或 (-1,0.5)·(-2,-1.5) = 1.25……
    best = max(sum(a * b for a, b in zip(qi, kj)) for qi in x for kj in [[2, -1], [-2, -1.5], [0, 3]]) / math.sqrt(2)
    check_close(max_logit(x, wq, wk), best, what="所有 (i, j) 上的最大 logit，除以 √d")


def test_only_large_heads_and_no_mutation():
    x = [[1.0, 0.5], [0.3, -1.0], [2.0, 0.2]]
    Wq = [[[1.0, 0.2], [0.1, 1.0]], [[30.0, 0.0], [0.0, 30.0]]]
    Wk = [[[0.5, 0.0], [0.0, 0.5]], [[20.0, 1.0], [1.0, 20.0]]]
    saved = copy.deepcopy((Wq, Wk))
    new_q, new_k, g = qk_clip(x, Wq, Wk, tau=50.0, alpha=0.3)
    check((Wq, Wk), saved, "不能修改传入的权重")
    check_close(g[0], 1.0, what="没超过阈值的头 γ = 1")
    check(new_q[0], Wq[0], "没超过阈值的头 W_q 不变")
    s = max_logit(x, Wq[1], Wk[1])
    check_close(g[1], 50.0 / s, what="超过阈值的头 γ = τ / S_max")
    check_close(max_logit(x, new_q[1], new_k[1]), 50.0, rtol=1e-6, what="缩放之后最大 logit 恰好等于 τ")
    check_close(new_q[1][0][0], 30.0 * g[1] ** 0.3, what="W_q 乘 γ^α（α = 0.3）")
    check_close(new_k[1][0][0], 20.0 * g[1] ** 0.7, what="W_k 乘 γ^(1-α)")
