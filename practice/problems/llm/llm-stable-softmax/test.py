import math

import numpy as np

from checker import check, check_close
from solution import log_softmax, logsumexp, softmax


def ref_lse(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    return np.apply_along_axis(lambda r: max(r) + math.log(sum(math.exp(v - max(r)) for v in r))
                               if np.isfinite(max(r)) else -math.inf, axis, x)


def test_example():
    x = np.array([1.0, 2.0, 3.0])
    check_close(softmax(x), np.exp(x) / np.exp(x).sum(), what="softmax([1, 2, 3])")
    check_close(logsumexp(x), math.log(math.exp(1) + math.exp(2) + math.exp(3)), what="logsumexp([1, 2, 3])")


def test_large_values():
    x = np.array([[1000.0, 1000.0], [-1000.0, 0.0]])
    check_close(softmax(x), [[0.5, 0.5], [0.0, 1.0]], what="大数输入的 softmax")
    check_close(logsumexp(x), [1000 + math.log(2), 0.0], what="大数输入的 logsumexp")
    check_close(log_softmax(np.array([1000.0, 0.0]))[1], -1000.0, what="log_softmax 的小概率项")


def test_neg_inf_mask():
    x = np.array([[0.0, -np.inf, 1.0], [-np.inf, -np.inf, -np.inf]])
    s = softmax(x)
    assert not np.isnan(s).any(), f"softmax 出现了 nan：{s}"
    check_close(s[0], [1 / (1 + math.e), 0.0, math.e / (1 + math.e)], what="带 -inf 的行")
    check_close(s[1], [0.0, 0.0, 0.0], what="整行 -inf")
    l = logsumexp(x)
    check(bool(np.isneginf(l[1])), True, "整行 -inf 的 logsumexp 应该是 -inf")


def test_axis_and_shapes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4, 5)) * 10
    for axis in [0, 1, -1]:
        check(softmax(x, axis).shape, x.shape, f"axis={axis} 时 softmax 的形状")
        check_close(softmax(x, axis).sum(axis=axis), np.ones(np.delete(x.shape, axis % 3)), what=f"axis={axis} 求和为 1")
        check_close(logsumexp(x, axis), ref_lse(x, axis), what=f"axis={axis} 的 logsumexp")
        check_close(log_softmax(x, axis), x - np.expand_dims(ref_lse(x, axis), axis), what=f"axis={axis} 的 log_softmax")


def test_float32_dtype():
    x = (np.arange(12, dtype=np.float32).reshape(3, 4) * 100)
    s = softmax(x)
    check(s.dtype, np.dtype(np.float32), "float32 输入的 softmax 结果类型")
    assert np.isfinite(s).all()
