import math

import numpy as np

from checker import check_close
from solution import chinchilla_optimal, fit_power_law, train_days


def test_example():
    n, d = chinchilla_optimal(5.76e23)
    check_close(n, math.sqrt(5.76e23 / 120), rtol=1e-12, what="N")
    check_close(d, 20 * n, rtol=1e-12, what="D")
    check_close(train_days(7e9, 2e12, 1024, 989, 0.4), 6 * 7e9 * 2e12 / (1024 * 989e12 * 0.4) / 86400, rtol=1e-12,
                what="训练天数")


def test_fit_exact_power_law():
    x = np.array([1e6, 1e7, 1e8, 1e9])
    y = 3.5 * x ** -0.076
    a, b = fit_power_law(x, y)
    check_close((a, b), (3.5, -0.076), rtol=1e-8, what="(a, b)")


def test_fit_with_noise():
    rng = np.random.default_rng(0)
    x = np.logspace(6, 10, 30)
    y = 2.0 * x ** 0.3 * np.exp(rng.normal(0, 0.01, 30))
    a, b = fit_power_law(x, y)
    lx, ly = np.log(x), np.log(y)
    B = ((lx - lx.mean()) * (ly - ly.mean())).sum() / ((lx - lx.mean()) ** 2).sum()
    A = math.exp(ly.mean() - B * lx.mean())
    check_close((a, b), (A, B), rtol=1e-8, what="最小二乘解")


def test_tokens_per_param():
    n, d = chinchilla_optimal(6e20, tokens_per_param=100)
    check_close(6 * n * d, 6e20, rtol=1e-12, what="6ND = C")
    check_close(d / n, 100, rtol=1e-12, what="D / N")


def test_train_days_scaling():
    base = train_days(1e9, 1e11, 8, 312, 0.5)
    check_close(train_days(1e9, 1e11, 16, 312, 0.5), base / 2, rtol=1e-12, what="卡数翻倍，时间减半")
    check_close(train_days(2e9, 1e11, 8, 312, 0.25), base * 4, rtol=1e-12, what="参数翻倍、MFU 减半，时间乘 4")
