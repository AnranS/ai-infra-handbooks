import warnings

import numpy as np

from checker import check, check_close
from solution import ffn_dim_for_params, silu, swiglu_ffn, swiglu_ffn_merged


def ref_silu(z):
    z = np.asarray(z, dtype=np.float64)
    return np.array([v / (1 + np.exp(-v)) if v > -500 else 0.0 for v in z.ravel()]).reshape(z.shape)


def weights(d=16, dff=40, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((5, d)), rng.standard_normal((d, dff)) * 0.3, rng.standard_normal((d, dff)) * 0.3,
            rng.standard_normal((dff, d)) * 0.3)


def test_example():
    check(ffn_dim_for_params(4096), 11008, "ffn_dim_for_params(4096)")
    check_close(silu(np.array([0.0, 1.0])), [0.0, 1 / (1 + np.exp(-1))], what="silu([0, 1])")


def test_silu_stable():
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = silu(z)
    check_close(y, [0.0, ref_silu([-50.0])[0], 0.0, 50.0, 1000.0], atol=1e-12, what="极端输入的 silu")


def test_ffn():
    x, g, u, dn = weights()
    want = (ref_silu(x @ g) * (x @ u)) @ dn
    check_close(swiglu_ffn(x, g, u, dn), want, rtol=1e-9, atol=1e-9, what="swiglu_ffn")


def test_merged_equals_separate():
    x, g, u, dn = weights(seed=1)
    merged = np.concatenate([g, u], axis=1)
    check_close(swiglu_ffn_merged(x, merged, dn), swiglu_ffn(x, g, u, dn), rtol=1e-12, atol=1e-12,
                what="合并权重的结果与分开计算相同")


def test_ffn_dims():
    check(ffn_dim_for_params(5120), 13824, "LLaMA-13B")
    check(ffn_dim_for_params(8192, multiple_of=4096), 24576, "multiple_of=4096")
    check(ffn_dim_for_params(768, multiple_of=1), 2048, "multiple_of=1")
