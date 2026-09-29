import gpusim as gs

BLOCK = 128
MAX_R = 32


@gs.kernel
def conv1d(t, inp, w, out, n, R):
    i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
    if i < n:
        acc = 0.0
        for j in range(-R, R + 1):          # 每个线程从全局内存读 2R+1 个元素
            if 0 <= i + j < n:
                acc += w[j + R] * inp[i + j]
        out[i] = acc
