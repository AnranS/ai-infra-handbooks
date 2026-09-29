import numpy as np

from checker import check, check_close
from solution import verify, verify_greedy


def test_example():
    q = np.array([[0.5, 0.5, 0.0], [0.2, 0.2, 0.6]])
    p = np.array([[0.25, 0.75, 0.0], [0.6, 0.2, 0.2], [0.1, 0.1, 0.8]])
    check(verify([0, 2], q, p, [0.4, 0.1], 0.0, 0.5), [0, 2, 2], "全部接受（u0 < 0.5、u1 < 1/3），再从 p2 采一个奖励 token")
    check(verify([0, 2], q, p, [0.6, 0.1], 0.3, 0.5), [1], "第 0 个被拒绝，残差只剩 token 1")
    check(verify([0, 2], q, p, [0.1, 0.9], 0.5, 0.5), [0, 0], "第 1 个被拒绝，残差 [0.4, 0, 0] 归一化后是 token 0")


def test_accept_when_target_prefers():
    q = np.array([[0.9, 0.1]])
    p = np.array([[0.1, 0.9], [0.5, 0.5]])
    check(verify([1], q, p, [0.99], 0.0, 0.7), [1, 1], "p >= q 时一定接受")


def test_greedy():
    check(verify_greedy([5, 6, 7], [5, 6, 9, 1]), [5, 6, 9], "第 2 个不一致")
    check(verify_greedy([5, 6, 7], [5, 6, 7, 1]), [5, 6, 7, 1], "全部一致，追加奖励 token")
    check(verify_greedy([], [4]), [4], "k = 0")


def test_distribution_preserved():
    """统计检验：第一个输出 token 的分布等于 p0"""
    rng = np.random.default_rng(0)
    V = 5
    q0 = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    p0 = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    p = np.stack([p0, np.full(V, 0.2)])
    counts = np.zeros(V)
    trials = 20000
    for _ in range(trials):
        t = int(rng.choice(V, p=q0))                   # 草稿 token 按 q0 采样
        out = verify([t], q0[None, :], p, [rng.random()], rng.random(), rng.random())
        counts[out[0]] += 1
    check_close(counts / trials, p0, atol=0.015, what="第一个输出 token 的经验分布")
