import gpusim as gs

TILE = 8


@gs.kernel
def gemm_tiled(t, A, B, C, M, N, K):
    row = t.blockIdx.y * TILE + t.threadIdx.y
    col = t.blockIdx.x * TILE + t.threadIdx.x
    if row < M and col < N:
        acc = 0.0
        for k in range(K):                     # 朴素写法：每个元素被读 N/M 次
            acc += A[row, k] * B[k, col]
        C[row, col] = acc
