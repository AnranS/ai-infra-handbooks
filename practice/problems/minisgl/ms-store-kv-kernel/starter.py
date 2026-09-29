import gpusim as gs


@gs.kernel
def store_cache_kernel(t, k, v, k_cache, v_cache, out_loc, n, row):
    i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x        # 一个线程搬一个 token
    if i < n:
        dst = out_loc[i]
        for j in range(row):
            k_cache[dst, j] = k[i, j]
            v_cache[dst, j] = v[i, j]


def store_cache(k, v, k_cache, v_cache, out_loc):
    n, row = k.shape
    return store_cache_kernel[gs.cdiv(n, 32), 32](k, v, k_cache, v_cache, out_loc, n, row)
