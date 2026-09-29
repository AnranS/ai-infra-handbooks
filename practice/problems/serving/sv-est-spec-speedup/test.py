import random

from checker import check, check_close
from solution import best_gamma, estimate_alpha, expected_tokens, speedup


def test_example():
    check_close(expected_tokens(0.8, 4), (1 - 0.8 ** 5) / 0.2, what="α=0.8、γ=4")
    check_close(speedup(0.8, 4, 0.05), (1 - 0.8 ** 5) / 0.2 / 1.2, what="加速比")
    check(best_gamma(0.8, 0.05), 8, "最佳草稿长度")
    check(best_gamma(0.3, 0.5), 0, "不值得投机")


def test_limits():
    check_close(expected_tokens(1.0, 4), 5, what="全部接受")
    check_close(expected_tokens(0.0, 4), 1, what="全部拒绝：只有验证得到的 1 个 token")
    check_close(expected_tokens(0.8, 200), 5, rtol=1e-9, what="γ 很大时趋近 1/(1-α)")
    check_close(speedup(0.9, 0, 0.1), 1, what="γ=0 就是普通 decode")
    check(best_gamma(0.5, 0.1), 2, "α=0.5、c=0.1")
    check(best_gamma(0.99, 0.0, max_gamma=5), 5, "草稿免费时越长越好")


def test_estimate_alpha():
    check_close(estimate_alpha([4, 4, 2, 0], 4), 10 / 12, what="10 次成功、2 次失败")
    check_close(estimate_alpha([], 4), 0.0, what="没有记录")
    check_close(estimate_alpha([3, 3, 3], 3), 1.0, what="全部接受")
    rng = random.Random(0)
    logs = []
    for _ in range(20000):
        n = 0
        while n < 5 and rng.random() < 0.7:
            n += 1
        logs.append(n)
    check_close(estimate_alpha(logs, 5), 0.7, rtol=0.02, what="从模拟日志里估计出 α≈0.7")
