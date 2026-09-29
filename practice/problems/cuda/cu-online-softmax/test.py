import numpy as np

import gpusim as gs
from checker import check_close
from solution import softmax_rows


def ref(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def run(x):
    rows, cols = x.shape
    dx, dy = gs.to_device(x, "x"), gs.empty((rows, cols), name="y")
    st = softmax_rows[rows, 128](dx, dy, rows, cols)
    return dy.copy_to_host(), st


def test_example():
    x = np.random.default_rng(0).standard_normal((4, 300)).astype(np.float32)
    got, _ = run(x)
    check_close(got, ref(x), rtol=1e-4, atol=1e-6, what="4×300")


def test_shapes_and_large_values():
    rng = np.random.default_rng(1)
    for rows, cols, scale in [(1, 1, 1.0), (3, 50, 1.0), (2, 128, 1.0), (2, 513, 1.0), (2, 200, 1000.0)]:
        x = (rng.standard_normal((rows, cols)) * scale).astype(np.float32)
        got, _ = run(x)
        check_close(got, ref(x), rtol=1e-4, atol=1e-6, what=f"{rows}×{cols}（数值尺度 {scale}）")


def test_two_passes():
    x = np.random.default_rng(2).standard_normal((3, 1000)).astype(np.float32)
    got, st = run(x)
    check_close(got, ref(x), rtol=1e-4, atol=1e-6, what="3×1000")
    reads = st.global_load_bytes / x.nbytes
    assert reads <= 2.0 + 1e-9, f"输入被读了 {reads:.1f} 遍，要求只读两遍（online softmax）"
