import math

import numpy as np

from checker import check, check_close
from solution import half_perm, rope_half, rope_interleaved


def ref_rotate(x, positions, base, pairs):
    T, H, dh = x.shape
    out = x.astype(np.float64).copy()
    for t in range(T):
        for i, (p0, p1) in enumerate(pairs):
            th = positions[t] * base ** (-2 * i / dh)
            c, s = math.cos(th), math.sin(th)
            a, b = x[t, :, p0], x[t, :, p1]
            out[t, :, p0] = a * c - b * s
            out[t, :, p1] = a * s + b * c
    return out


def setup(T=5, H=3, dh=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, H, dh)), rng.integers(0, 1000, T)


def test_example():
    x, pos = setup()
    dh = 8
    check_close(rope_half(x, pos), ref_rotate(x, pos, 10000.0, [(i, i + dh // 2) for i in range(dh // 2)]),
                rtol=1e-9, atol=1e-9, what="rope_half")
    check_close(rope_interleaved(x, pos), ref_rotate(x, pos, 10000.0, [(2 * i, 2 * i + 1) for i in range(dh // 2)]),
                rtol=1e-9, atol=1e-9, what="rope_interleaved")


def test_position_zero_is_identity_and_norm_preserved():
    x, _ = setup(seed=1)
    check_close(rope_half(x, np.zeros(5, dtype=int)), x, atol=1e-12, what="位置 0 不旋转")
    y = rope_half(x, np.arange(5) * 37, base=1e6)
    check_close(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-9, what="旋转不改变长度")


def test_relative_position_property():
    rng = np.random.default_rng(2)
    q, k = rng.standard_normal((1, 1, 16)), rng.standard_normal((1, 1, 16))
    for f in (rope_half, rope_interleaved):
        dots = [float((f(q, np.array([m])) * f(k, np.array([n]))).sum()) for m, n in [(10, 3), (107, 100), (7, 0)]]
        check_close(dots, [dots[0]] * 3, rtol=1e-9, atol=1e-9, what=f"{f.__name__}：点积只取决于 m - n")


def test_perm_converts_between_layouts():
    for dh in [2, 8, 64]:
        x, pos = setup(T=4, H=2, dh=dh, seed=dh)
        perm = half_perm(dh)
        check(sorted(perm.tolist()), list(range(dh)), f"d_h={dh} 时 perm 是一个排列")
        check_close(rope_half(x[..., perm], pos), rope_interleaved(x, pos)[..., perm], rtol=1e-9, atol=1e-9,
                    what=f"d_h={dh} 时两种写法经过重排后一致")


def test_custom_base():
    x, pos = setup(seed=5)
    dh = 8
    check_close(rope_half(x, pos, base=500000.0),
                ref_rotate(x, pos, 500000.0, [(i, i + dh // 2) for i in range(dh // 2)]), rtol=1e-9, atol=1e-9,
                what="base=500000（LLaMA 3）")
