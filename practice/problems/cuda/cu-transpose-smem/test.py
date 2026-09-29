import numpy as np

import gpusim as gs
from checker import check
from solution import transpose_kernel


def run(rows, cols, seed=0):
    a = np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)
    da, db = gs.to_device(a, "a"), gs.empty((cols, rows), name="b")
    stats = transpose_kernel[(gs.cdiv(cols, 32), gs.cdiv(rows, 32)), (32, 8)](da, db, rows, cols)
    return a, db.copy_to_host(), stats


def test_example_square():
    """64×64：结果正确"""
    a, b, _ = run(64, 64)
    check(b, a.T, "转置结果")


def test_odd_shape():
    """70×45：行、列都不是 32 的倍数"""
    a, b, _ = run(70, 45, seed=1)
    check(b, a.T, "转置结果")


def test_thin():
    """3×100 和 100×3"""
    for shape in [(3, 100), (100, 3)]:
        a, b, _ = run(*shape, seed=2)
        check(b, a.T, f"{shape} 的转置结果")


def test_coalesced():
    """64×64：全局内存读写都完全合并"""
    _, _, st = run(64, 64)
    assert st.load_efficiency == 1.0, f"读效率只有 {st.load_efficiency:.0%}：{st}"
    assert st.store_efficiency == 1.0, f"写效率只有 {st.store_efficiency:.0%}（写是跨步的？）：{st}"


def test_no_bank_conflict():
    """64×64：共享内存没有 bank conflict"""
    _, _, st = run(64, 64)
    assert st.shared_load_requests > 0, "没有用到共享内存"
    assert st.bank_conflicts == 0, f"有 {st.bank_conflicts} 次 bank conflict：{st}"
