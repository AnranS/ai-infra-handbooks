import math

import numpy as np

from checker import check, check_close
from solution import flashinfer_metadata, paged_attention_from_meta


def build(page_size, lens, num_pages=32, seed=0):
    rng = np.random.default_rng(seed)
    pt = np.full((len(lens), 64), -1, dtype=np.int32)
    perm = list(rng.permutation(num_pages))
    for r, L in enumerate(lens):
        for j in range(math.ceil(L / page_size)):
            pg = perm.pop()
            pt[r, j * page_size:(j + 1) * page_size] = pg * page_size + np.arange(page_size)
    return pt


def test_example():
    pt = build(4, [6, 9])
    reqs = [(0, 6, 6), (1, 1, 9)]
    m = flashinfer_metadata(reqs, pt, 4)
    check(m["qo_indptr"].tolist(), [0, 6, 7], "qo_indptr")
    check(m["kv_indptr"].tolist(), [0, 2, 5], "kv_indptr")
    check(m["kv_indices"].tolist(), [pt[0, 0] // 4, pt[0, 4] // 4, pt[1, 0] // 4, pt[1, 4] // 4, pt[1, 8] // 4],
          "kv_indices")
    check(m["kv_last_page_len"].tolist(), [2, 1], "kv_last_page_len")


def test_page_size_one_like_minisgl():
    pt = build(1, [3, 5])
    m = flashinfer_metadata([(0, 3, 3), (1, 1, 5)], pt, 1)
    check(m["kv_last_page_len"].tolist(), [1, 1], "page_size=1 时 last_page_len 恒为 1")
    check(m["kv_indices"].tolist(), pt[0, :3].tolist() + pt[1, :5].tolist(), "kv_indices 就是 token 位置")


def ref(q, kc, vc, reqs, pt, ps):
    out = []
    H, d = q.shape[1], q.shape[2]
    start = 0
    for row, n, L in reqs:
        ks = np.stack([kc[pt[row, p] // ps, pt[row, p] % ps] for p in range(L)])
        vs = np.stack([vc[pt[row, p] // ps, pt[row, p] % ps] for p in range(L)])
        for j in range(n):
            pos = L - n + j
            s = np.einsum("hd,shd->hs", q[start + j], ks[:pos + 1]) / math.sqrt(d)
            w = np.exp(s - s.max(1, keepdims=True))
            out.append(np.einsum("hs,shd->hd", w / w.sum(1, keepdims=True), vs[:pos + 1]))
        start += n
    return np.stack(out)


def test_attention_with_meta():
    rng = np.random.default_rng(1)
    for ps in [1, 4, 8]:
        lens = [5, 16, 9, 1]
        pt = build(ps, lens, num_pages=64 // ps + 8, seed=ps)
        reqs = [(0, 5, 5), (1, 3, 16), (2, 1, 9), (3, 1, 1)]
        m = flashinfer_metadata(reqs, pt, ps)
        npages = pt.max() // ps + 1
        kc, vc = rng.standard_normal((npages, ps, 2, 4)), rng.standard_normal((npages, ps, 2, 4))
        q = rng.standard_normal((10, 2, 4))
        check_close(paged_attention_from_meta(q, kc, vc, m, ps), ref(q, kc, vc, reqs, pt, ps), rtol=1e-9, atol=1e-10,
                    what=f"page_size={ps} 的注意力")
