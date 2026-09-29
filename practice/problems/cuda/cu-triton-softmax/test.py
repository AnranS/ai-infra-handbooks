import numpy as np

import tritonkit
from checker import check, check_close
from solution import softmax


def ref(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def run(x):
    return tritonkit.to_host(softmax(tritonkit.to_dev(x)))


def test_example():
    x = np.random.default_rng(0).standard_normal((4, 128)).astype(np.float32)
    check_close(run(x), ref(x), rtol=1e-5, atol=1e-6, what="4×128")


def test_non_power_of_two():
    for rows, cols in [(1, 1), (3, 100), (5, 781), (2, 1000)]:
        x = np.random.default_rng(cols).standard_normal((rows, cols)).astype(np.float32)
        check_close(run(x), ref(x), rtol=1e-5, atol=1e-6, what=f"{rows}×{cols}")


def test_large_values():
    x = (np.random.default_rng(1).standard_normal((3, 50)) * 1000).astype(np.float32)
    got = run(x)
    assert np.isfinite(got).all(), "大输入时出现了 inf/nan：要先减去最大值"
    check_close(got, ref(x), rtol=1e-4, atol=1e-6, what="大输入")


def test_fused_single_pass():
    x = np.random.default_rng(2).standard_normal((6, 300)).astype(np.float32)
    run(x)
    st = tritonkit.last_stats()
    if st is None:
        return                               # 真 Triton 下没有这个统计
    check((st.programs, st.loads, st.stores), (6, 6, 6), "每行一个 program、一次 load、一次 store")
    check(st.load_bytes, x.nbytes, "输入只读一遍")
