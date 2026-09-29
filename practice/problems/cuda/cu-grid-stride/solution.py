import gpusim as gs


@gs.kernel
def saxpy(t, a, x, y, n):
    i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
    stride = t.gridDim.x * t.blockDim.x
    while i < n:
        y[i] = a * x[i] + y[i]
        i += stride
