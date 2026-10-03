import numpy as np

from checker import check, check_close
from solution import lora_params, low_rank, rank_for_energy


def test_example():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 32)) @ rng.standard_normal((32, 48))
    B, A = low_rank(W, 8)
    check((B.shape, A.shape), ((64, 8), (8, 48)), "B、A 的形状")
    check(lora_params([(4096, 4096)] * 4, r=16), (67108864, 524288, 0.0078125), "lora_params")


def test_optimal_error():
    """误差等于被丢掉的奇异值的平方和（Eckart–Young）"""
    rng = np.random.default_rng(1)
    W = rng.standard_normal((40, 30))
    S = np.linalg.svd(W, compute_uv=False)
    for r in [1, 5, 29, 30]:
        B, A = low_rank(W, r)
        err = np.linalg.norm(W - B @ A) ** 2
        check_close(err, np.sum(S[r:] ** 2), rtol=1e-8, atol=1e-8, what=f"r={r} 的近似误差")
        check(np.linalg.matrix_rank(B @ A), min(r, 30), f"r={r} 时 BA 的秩")


def test_balanced_factors():
    """奇异值平均分到两边：B 的列范数平方 = A 的行范数平方 = 奇异值"""
    rng = np.random.default_rng(2)
    W = rng.standard_normal((20, 50))
    B, A = low_rank(W, 6)
    S = np.linalg.svd(W, compute_uv=False)[:6]
    check_close(np.sum(B ** 2, axis=0), S, rtol=1e-8, what="B 每列的范数平方")
    check_close(np.sum(A ** 2, axis=1), S, rtol=1e-8, what="A 每行的范数平方")


def test_exact_low_rank_matrix():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((50, 4)) @ rng.standard_normal((4, 60))
    B, A = low_rank(W, 4)
    check_close(B @ A, W, rtol=1e-8, atol=1e-8, what="秩 4 的矩阵用 r=4 完全重建")
    check(rank_for_energy(W, 1.0), 4, "rank_for_energy(W, 1.0)")


def test_energy():
    W = np.diag([3.0, 2.0, 1.0])          # 能量 9, 4, 1，共 14
    check(rank_for_energy(W, 0.5), 1, "9/14 ≥ 0.5")
    check(rank_for_energy(W, 0.9), 2, "13/14 ≥ 0.9")
    check(rank_for_energy(W, 0.95), 3, "需要全部")
    check(rank_for_energy(W, 9 / 14), 1, "正好等于边界")


def test_lora_params_mixed():
    shapes = [(1024, 1024), (1024, 3072), (3072, 1024)]
    full, lora, ratio = lora_params(shapes, 8)
    check(full, 1024 * 1024 + 2 * 1024 * 3072, "原参数量")
    check(lora, 8 * (2048 + 4096 + 4096), "LoRA 参数量")
    check_close(ratio, lora / full, what="比例")
