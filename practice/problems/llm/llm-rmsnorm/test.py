import numpy as np

from checker import check, check_close
from solution import fused_add_rms_norm, layer_norm, pre_norm_block, rms_norm


def ref_rms(x, w, eps=1e-6):
    x = x.astype(np.float64)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w


def test_example():
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    check_close(rms_norm(x, w), x / np.sqrt(7.5 + 1e-6), rtol=1e-6, what="rms_norm")


def test_rms_norm_shapes_and_weight():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 16)).astype(np.float32)
    w = rng.standard_normal(16).astype(np.float32)
    y = rms_norm(x, w)
    check((y.shape, y.dtype), (x.shape, np.dtype(np.float32)), "形状和类型")
    check_close(y, ref_rms(x, w), rtol=1e-5, atol=1e-6, what="rms_norm")


def test_layer_norm():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 32)).astype(np.float32) * 3 + 1
    w, b = rng.standard_normal(32).astype(np.float32), rng.standard_normal(32).astype(np.float32)
    x64 = x.astype(np.float64)
    want = (x64 - x64.mean(-1, keepdims=True)) / np.sqrt(x64.var(-1, keepdims=True) + 1e-5) * w + b
    check_close(layer_norm(x, w, b), want, rtol=1e-5, atol=1e-5, what="layer_norm")


def test_float16_no_overflow():
    x = np.full((1, 8), 300.0, dtype=np.float16)          # 300² 超过 float16 的最大值 65504
    y = rms_norm(x, np.ones(8, dtype=np.float16))
    check(y.dtype, np.dtype(np.float16), "float16 输入返回 float16")
    check_close(y.astype(np.float32), np.ones((1, 8)), rtol=1e-3, what="在 float32 里算才不会溢出")


def test_pre_norm_block():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal(8).astype(np.float32)
    W = rng.standard_normal((8, 8)).astype(np.float32)
    out = pre_norm_block(x, lambda h: h @ W, w)
    check_close(out, x + ref_rms(x, w) @ W, rtol=1e-4, atol=1e-5, what="x + sublayer(norm(x))")


def test_fused_add_rms_norm():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 16)).astype(np.float16)
    r = rng.standard_normal((3, 16)).astype(np.float16)
    w = rng.standard_normal(16).astype(np.float16)
    normed, res = fused_add_rms_norm(x, r, w)
    check((normed.dtype, res.dtype), (np.dtype(np.float16), np.dtype(np.float16)), "返回值类型")
    want_res = (x.astype(np.float32) + r.astype(np.float32))
    check_close(res.astype(np.float32), want_res, rtol=1e-3, atol=1e-3, what="新的 residual")
    check_close(normed.astype(np.float32), ref_rms(res, w.astype(np.float32)), rtol=2e-3, atol=2e-3, what="normed")
