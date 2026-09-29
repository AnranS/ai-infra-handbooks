import gpusim as gs


@gs.kernel
def whoami(t, out):
    tid = t.threadIdx.y * t.blockDim.x + t.threadIdx.x
    bid = t.blockIdx.y * t.gridDim.x + t.blockIdx.x
    gid = bid * (t.blockDim.x * t.blockDim.y) + tid
    out[gid, 0] = gid
    out[gid, 1] = bid
    out[gid, 2] = tid // 32
    out[gid, 3] = tid % 32
