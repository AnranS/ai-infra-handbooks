import numpy as np

import gpusim as gs
from checker import check
from solution import store_cache


def run(n, row, slots, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((n, row)).astype(np.float32)
    v = rng.standard_normal((n, row)).astype(np.float32)
    kc0 = rng.standard_normal((slots, row)).astype(np.float32)
    vc0 = rng.standard_normal((slots, row)).astype(np.float32)
    loc = rng.permutation(slots)[:n].astype(np.int32)
    dk, dv = gs.to_device(k, "k"), gs.to_device(v, "v")
    dkc, dvc = gs.to_device(kc0, "k_cache"), gs.to_device(vc0, "v_cache")
    st = store_cache(dk, dv, dkc, dvc, gs.to_device(loc, "out_loc"))
    wk, wv = kc0.copy(), vc0.copy()
    wk[loc], wv[loc] = k, v
    return dkc.copy_to_host(), dvc.copy_to_host(), wk, wv, st


def test_example():
    kc, vc, wk, wv, _ = run(5, 64, 16)
    check(kc, wk, "k_cache")
    check(vc, wv, "v_cache")


def test_sizes():
    for n, row, slots in [(1, 1, 3), (7, 100, 20), (3, 257, 5)]:
        kc, vc, wk, wv, _ = run(n, row, slots, seed=n)
        check(kc, wk, f"n={n}, row={row} 的 k_cache")
        check(vc, wv, f"n={n}, row={row} 的 v_cache")


def test_coalesced():
    kc, vc, wk, wv, st = run(8, 256, 32, seed=5)
    check(kc, wk, "结果")
    for name, key in [("k", "load_efficiency"), ("k_cache", "store_efficiency")]:
        eff = st.array(name)[key]
        assert eff >= 0.9, f"{name} 的访存效率只有 {eff:.0%}，要求 ≥ 90%"
