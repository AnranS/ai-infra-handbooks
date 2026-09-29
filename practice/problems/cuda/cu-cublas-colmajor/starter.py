import numpy as np


def _view(buf, rows, cols, ld):
    """列主序缓冲区 buf 上 rows × cols 的矩阵视图（元素 (i, j) 在 buf[i + j * ld]）。"""
    if ld < max(1, rows):
        raise ValueError(f"leading dimension {ld} 小于行数 {rows}")
    if rows and cols and (rows - 1) + (cols - 1) * ld >= buf.size:
        raise ValueError("缓冲区太小")
    return np.lib.stride_tricks.as_strided(buf, shape=(rows, cols), strides=(buf.itemsize, ld * buf.itemsize))


def sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """模拟 cublasSgemm：C = alpha * op(A) @ op(B) + beta * C，所有矩阵都是列主序。"""
    opA = _view(A, m, k, lda) if transa == "N" else _view(A, k, m, lda).T
    opB = _view(B, k, n, ldb) if transb == "N" else _view(B, n, k, ldb).T
    Cv = _view(C, m, n, ldc)
    Cv[...] = alpha * (opA.astype(np.float64) @ opB.astype(np.float64)) + (beta * Cv if beta else 0)


def matmul(A, B):
    M, K = A.shape
    _, N = B.shape
    C = np.empty(M * N, dtype=np.float32)
    # 直接照搬公式：把行主序当成列主序传进去，结果是错的
    sgemm("N", "N", M, N, K, 1.0, A.reshape(-1), M, B.reshape(-1), K, 0.0, C, M)
    return C.reshape(M, N)


def matmul_at_b(A, B):
    pass
