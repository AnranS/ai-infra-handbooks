"""CUDA 模拟器：朴素转置与共享内存转置的访存对比"""
import numpy as np

import gpusim as gs

N, TILE = 64, 32


@gs.kernel
def naive(t, a, out):
    x = t.blockIdx.x * TILE + t.threadIdx.x
    y = t.blockIdx.y * TILE + t.threadIdx.y
    out[x * N + y] = a[y * N + x]                          # 读是合并的，写的地址跨度是 N：不合并


@gs.kernel
def tiled(t, a, out):
    s = t.shared("s", (TILE, TILE + 1))                    # 多出的一列避免 bank conflict
    x = t.blockIdx.x * TILE + t.threadIdx.x
    y = t.blockIdx.y * TILE + t.threadIdx.y
    s[t.threadIdx.y, t.threadIdx.x] = a[y * N + x]
    yield t.syncthreads()
    x2 = t.blockIdx.y * TILE + t.threadIdx.x
    y2 = t.blockIdx.x * TILE + t.threadIdx.y
    out[y2 * N + x2] = s[t.threadIdx.x, t.threadIdx.y]     # 读共享内存换个方向，写全局内存就是合并的


a = np.arange(N * N, dtype=np.float32)
for kernel in (naive, tiled):
    src, dst = gs.to_device(a, "a"), gs.empty(N * N, name="out")
    stats = kernel[(N // TILE, N // TILE), (TILE, TILE)](src, dst)
    ok = np.array_equal(dst.copy_to_host().reshape(N, N), a.reshape(N, N).T)
    print(f"{kernel.__name__:<6} 结果正确 {ok}；写全局内存 {stats.global_store_sectors} 个扇区，写效率 {stats.store_efficiency:.0%}，bank conflict {stats.bank_conflicts}")
