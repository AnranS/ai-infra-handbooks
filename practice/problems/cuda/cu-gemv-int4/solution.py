import gpusim as gs


def warp_sum(t, v):
    off = 16
    while off > 0:
        v += yield t.shfl_down(v, off)
        off //= 2
    return v


@gs.kernel
def gemv_w4(t, packed, scales, x, y, N, K):
    lane = t.threadIdx.x
    row = t.blockIdx.x * 4 + t.threadIdx.y
    acc = 0.0
    if row < N:
        for j in range(lane, K // 2, 32):
            b = int(packed[row, j])
            acc += ((b & 0xF) - 8) * x[2 * j] + ((b >> 4) - 8) * x[2 * j + 1]
    s = yield from warp_sum(t, acc)
    if lane == 0 and row < N:
        y[row] = s * scales[row]
