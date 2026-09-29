import gpusim as gs

BLOCK = 128
MAX_R = 32


@gs.kernel
def conv1d(t, inp, w, out, n, R):
    tile = t.shared("tile", BLOCK + 2 * MAX_R)
    ws = t.shared("w", 2 * MAX_R + 1)
    tx = t.threadIdx.x
    base = t.blockIdx.x * BLOCK - R
    for k in range(tx, BLOCK + 2 * R, BLOCK):
        g = base + k
        tile[k] = inp[g] if 0 <= g < n else 0.0
    if tx < 2 * R + 1:
        ws[tx] = w[tx]
    yield t.syncthreads()
    i = t.blockIdx.x * BLOCK + tx
    if i < n:
        acc = 0.0
        for j in range(2 * R + 1):
            acc += ws[j] * tile[tx + j]
        out[i] = acc
