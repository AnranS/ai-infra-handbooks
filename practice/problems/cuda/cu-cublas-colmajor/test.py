import numpy as np

import solution
from checker import check, check_close

CALLS = []


def _view(buf, rows, cols, ld):
    if ld < max(1, rows):
        raise ValueError(f"leading dimension {ld} 小于行数 {rows}")
    if rows and cols and (rows - 1) + (cols - 1) * ld >= buf.size:
        raise ValueError("缓冲区太小")
    return np.lib.stride_tricks.as_strided(buf, shape=(rows, cols), strides=(buf.itemsize, ld * buf.itemsize))


def ref_sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    CALLS.append({"transa": transa, "transb": transb, "A": A, "B": B})
    opA = _view(A, m, k, lda) if transa == "N" else _view(A, k, m, lda).T
    opB = _view(B, k, n, ldb) if transb == "N" else _view(B, n, k, ldb).T
    Cv = _view(C, m, n, ldc)
    Cv[...] = alpha * (opA.astype(np.float64) @ opB.astype(np.float64)) + (beta * Cv if beta else 0)


def call(fn, A, B):
    CALLS.clear()
    orig = solution.sgemm
    solution.sgemm = ref_sgemm
    try:
        C = fn(A, B)
    finally:
        solution.sgemm = orig
    return C


def rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def check_buffers(A, B):
    check(len(CALLS), 1, "sgemm 的调用次数")
    bufs = [CALLS[0]["A"], CALLS[0]["B"]]
    assert any(np.shares_memory(b, A) for b in bufs) and any(np.shares_memory(b, B) for b in bufs), \
        "传给 sgemm 的缓冲区应该直接是输入的视图（不能先转置或复制）"


def test_example():
    A, B = rand(3, 5), rand(5, 4, seed=1)
    C = call(solution.matmul, A, B)
    check(C.shape, (3, 4), "形状")
    check_close(C, A.astype(np.float64) @ B, rtol=1e-5, atol=1e-5, what="A @ B")
    check_buffers(A, B)


def test_matmul_shapes():
    for M, K, N in [(1, 1, 1), (7, 3, 9), (16, 32, 8)]:
        A, B = rand(M, K, seed=M), rand(K, N, seed=N)
        check_close(call(solution.matmul, A, B), A.astype(np.float64) @ B, rtol=1e-5, atol=1e-5, what=f"{M}×{K} @ {K}×{N}")


def test_at_b():
    for K, M, N in [(5, 3, 4), (8, 1, 6), (10, 7, 7)]:
        A, B = rand(K, M, seed=K), rand(K, N, seed=M)
        C = call(solution.matmul_at_b, A, B)
        check_close(C, A.T.astype(np.float64) @ B, rtol=1e-5, atol=1e-5, what=f"A^T @ B（A: {K}×{M}）")
        check_buffers(A, B)
