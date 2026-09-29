import gpusim as gs


@gs.kernel
def whoami(t, out):
    gid = t.blockIdx.x * t.blockDim.x + t.threadIdx.x     # 只考虑了一维
    out[gid, 0] = gid
