import gpusim as gs


def warp_sum(t, v):
    off = 16
    while off > 0:
        v += yield t.shfl_down(v, off)
        off //= 2
    return v


@gs.kernel
def gemv_w4(t, packed, scales, x, y, N, K):
    # 一个线程算一行：相邻线程读的地址相隔 K/2 字节，不合并
    row = (t.blockIdx.x * 4 + t.threadIdx.y) * 32 + t.threadIdx.x
    if row < N:
        acc = 0.0
        for j in range(K // 2):
            b = int(packed[row, j])
            acc += ((b & 0xF) - 8) * x[2 * j] + ((b >> 4) - 8) * x[2 * j + 1]
        y[row] = acc * scales[row]
