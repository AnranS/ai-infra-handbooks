import gpusim as gs

TILE = 8


@gs.kernel
def gemm_tiled(t, A, B, C, M, N, K):
    As = t.shared("As", (TILE, TILE))
    Bs = t.shared("Bs", (TILE, TILE))
    tx, ty = t.threadIdx.x, t.threadIdx.y
    row = t.blockIdx.y * TILE + ty
    col = t.blockIdx.x * TILE + tx
    acc = 0.0
    for k0 in range(0, K, TILE):
        As[ty, tx] = A[row, k0 + tx] if row < M and k0 + tx < K else 0.0
        Bs[ty, tx] = B[k0 + ty, col] if k0 + ty < K and col < N else 0.0
        yield t.syncthreads()
        for k in range(TILE):
            acc += As[ty, k] * Bs[k, tx]
        yield t.syncthreads()
    if row < M and col < N:
        C[row, col] = acc
