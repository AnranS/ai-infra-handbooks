import numpy as np

from checker import check, check_close
from solution import fused_moe_grouped, moe_align_block_size


def test_example():
    topk_ids = np.array([[2, 0], [0, 2], [2, 1]])     # pair 下标 0..5 对应的专家：2 0 0 2 2 1
    sorted_ids, expert_ids = moe_align_block_size(topk_ids, block_m=2, num_experts=4)
    check(sorted_ids.tolist(), [1, 2, 5, 6, 0, 3, 4, 6], "专家 0：pair 1、2；专家 1：pair 5 + 补齐；专家 2：pair 0、3、4 + 补齐")
    check(expert_ids.tolist(), [0, 1, 2, 2], "每个块的专家")


def ref_moe(x, w, topk_ids, topk_w):
    T, k = topk_ids.shape
    out = np.zeros((T, w.shape[1]))
    for t in range(T):
        for j in range(k):
            out[t] += topk_w[t, j] * (w[topk_ids[t, j]] @ x[t])
    return out


def test_align_random_properties():
    rng = np.random.default_rng(0)
    for trial in range(30):
        T, k, E, bm = rng.integers(1, 40), rng.integers(1, 4), rng.integers(1, 9), int(rng.choice([1, 4, 16]))
        ids = np.stack([rng.choice(E, k, replace=False) if k <= E else rng.integers(0, E, k) for _ in range(T)])
        sorted_ids, expert_ids = moe_align_block_size(ids, bm, E)
        flat = ids.ravel()
        counts = np.bincount(flat, minlength=E)
        check(len(sorted_ids), int(sum(-(-c // bm) * bm for c in counts)), f"随机用例 {trial}：总长度")
        check(len(expert_ids) * bm, len(sorted_ids), f"随机用例 {trial}：块数")
        for b, e in enumerate(expert_ids):
            blk = sorted_ids[b * bm:(b + 1) * bm]
            valid = blk[blk < flat.size]
            assert all(flat[valid] == e), f"随机用例 {trial}：第 {b} 块里混进了别的专家"
        valid = sorted_ids[sorted_ids < flat.size]
        check(sorted(valid.tolist()), list(range(flat.size)), f"随机用例 {trial}：每个 pair 恰好出现一次")
        for e in range(E):
            seg = [i for i in valid.tolist() if flat[i] == e]
            check(seg, sorted(seg), f"随机用例 {trial}：专家 {e} 的段内保持原顺序")


def test_grouped_gemm():
    rng = np.random.default_rng(1)
    T, k, E, din, dout = 20, 2, 5, 6, 4
    x = rng.standard_normal((T, din))
    w = rng.standard_normal((E, dout, din))
    ids = np.stack([rng.choice(E, k, replace=False) for _ in range(T)])
    tw = rng.dirichlet(np.ones(k), T)
    for bm in [1, 4, 16]:
        check_close(fused_moe_grouped(x, w, ids, tw, bm), ref_moe(x, w, ids, tw), rtol=1e-10, atol=1e-12,
                    what=f"block_m={bm} 的分组 GEMM")
