import math

import numpy as np

from checker import check, check_close
from solution import mha


def ref(x, wq, wk, wv, wo, H, causal=True, mask=None):
    B, T, d = x.shape
    dh = d // H
    out = np.zeros((B, T, d))
    for b in range(B):
        q, k, v = x[b] @ wq, x[b] @ wk, x[b] @ wv
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            for i in range(T):
                js = [j for j in range(T) if (not causal or j <= i) and (mask is None or mask[b, j])]
                if not js:
                    continue
                s = [q[i, sl] @ k[j, sl] / math.sqrt(dh) for j in js]
                m = max(s)
                w = [math.exp(v_ - m) for v_ in s]
                tot = sum(w)
                out[b, i, sl] = sum(wi / tot * v[j, sl] for wi, j in zip(w, js))
    return out @ wo


def setup(B=2, T=5, d=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, T, d))
    ws = [rng.standard_normal((d, d)) * 0.5 for _ in range(4)]
    return x, ws


def test_example():
    x, ws = setup()
    check_close(mha(x, *ws, n_heads=2), ref(x, *ws, 2), rtol=1e-9, atol=1e-9, what="因果多头注意力")


def test_non_causal_and_heads():
    x, ws = setup(seed=1)
    for H in [1, 4, 8]:
        check_close(mha(x, *ws, n_heads=H, causal=False), ref(x, *ws, H, causal=False), rtol=1e-9, atol=1e-9,
                    what=f"非因果，{H} 个头")


def test_first_position_sees_only_itself():
    x, ws = setup(seed=2)
    out = mha(x, *ws, n_heads=2)
    check_close(out[:, 0], (x[:, 0] @ ws[2]) @ ws[3], rtol=1e-9, atol=1e-9, what="位置 0 的输出就是它自己的 V")


def test_padding_mask():
    x, ws = setup(B=3, T=6, seed=3)
    mask = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1]], dtype=bool)
    got = mha(x, *ws, n_heads=4, mask=mask)
    assert not np.isnan(got).any(), "输出里有 nan：一个 key 都看不到的 query 应该输出 0"
    check_close(got, ref(x, *ws, 4, mask=mask), rtol=1e-9, atol=1e-9, what="带 padding 掩码")
    check_close(got[2, :2], np.zeros((2, 8)), atol=1e-12, what="左侧 padding 的位置看不到任何 key，输出 0")


def test_padding_does_not_leak():
    """改变 padding 位置上的输入，不影响真实位置的输出"""
    x, ws = setup(B=1, T=6, seed=4)
    mask = np.array([[1, 1, 1, 1, 0, 0]], dtype=bool)
    a = mha(x, *ws, n_heads=2, causal=False, mask=mask)
    x2 = x.copy()
    x2[0, 4:] = 1000.0
    b = mha(x2, *ws, n_heads=2, causal=False, mask=mask)
    check_close(a[0, :4], b[0, :4], rtol=1e-9, atol=1e-9, what="真实位置的输出")
