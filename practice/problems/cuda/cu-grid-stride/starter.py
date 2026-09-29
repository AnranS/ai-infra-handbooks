import gpusim as gs


@gs.kernel
def saxpy(t, a, x, y, n):
    i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
    if i < n:
        y[i] = a * x[i] + y[i]          # 线程数少于 n 时，后面的元素没人处理
