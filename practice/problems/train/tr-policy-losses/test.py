import math

from checker import check, check_close, raises
from solution import cispo_weights, gspo_ratio, k3, token_weights


def close_all(actual, expected, what, **tol):
    check(len(actual), len(expected), f"{what}：个数")
    for i, (a, e) in enumerate(zip(actual, expected)):
        check_close(a, e, what=f"{what}（第 {i} 个）", **tol)


def test_example():
    close_all(token_weights([1.0, -1.0], [2, 4], "seq"), [0.25, -0.125], "按回答平均")
    close_all(token_weights([1.0, -1.0], [2, 4], "token"), [1 / 6, -1 / 6], "按 token 平均")
    check_close(gspo_ratio([-1.0, -2.0], [-1.2, -1.8]), 1.0, what="几何平均")


def test_weights_sum():
    adv, lens = [0.87, -0.87, -0.87, 0.87], [200, 50, 2000, 400]
    seq, tok = token_weights(adv, lens, "seq"), token_weights(adv, lens, "token")
    close_all([w * L for w, L in zip(seq, lens)], [0.2175, -0.2175, -0.2175, 0.2175], "按回答平均：每个回答的总权重相同")
    check_close(tok[2] * lens[2] / (tok[1] * lens[1]), 40.0, what="按 token 平均：2000 个 token 的回答总权重是 50 个的 40 倍")
    with raises(ValueError):
        token_weights([1.0], [3], "mean")


def test_gspo():
    new = [-0.5, -1.0, -2.0, -0.1]
    old = [-0.6, -1.3, -1.9, -0.1]
    check_close(gspo_ratio(new, old), math.exp(0.3 / 4), what="log 比率之和 0.3，长度 4")
    long_new, long_old = [-1.0] * 1000, [-1.01] * 1000
    check_close(gspo_ratio(long_new, long_old), math.exp(0.01), what="1000 个 token 每个都高 0.01：几何平均 e^0.01，不是乘积 e^10")


def test_cispo_and_k3():
    close_all(cispo_weights([0.5, 0.9, 1.1, 1.5, 3.0]), [0.8, 0.9, 1.1, 1.28, 1.28], "截断到 [0.8, 1.28]")
    close_all(cispo_weights([0.5, 1.5], eps_low=0.1, eps_high=0.1), [0.9, 1.1], "自定义的截断范围")
    vals = k3([-1.0, -2.0, -0.5], [-1.0, -1.5, -1.2])
    check_close(vals[0], 0.0, atol=1e-12, what="两个分布在这个 token 上一致时 k3 = 0")
    check_close(vals[1], math.exp(0.5) - 1 - 0.5, what="log r = 0.5")
    check_close(vals[2], math.exp(-0.7) - 1 + 0.7, what="log r = -0.7")
    if min(vals) < 0:
        raise AssertionError("k3 不应为负")
