import gpusim as gs


@gs.kernel
def store_cache_kernel(t, k, v, k_cache, v_cache, out_loc, n, row):
    i = t.blockIdx.x                                      # 一个 block 负责一个 token
    dst = out_loc[i]
    for j in range(t.threadIdx.x, row, t.blockDim.x):     # 线程沿着这一行跨步搬运：合并访问
        k_cache[dst, j] = k[i, j]
        v_cache[dst, j] = v[i, j]


def store_cache(k, v, k_cache, v_cache, out_loc):
    n, row = k.shape
    return store_cache_kernel[n, 64](k, v, k_cache, v_cache, out_loc, n, row)
