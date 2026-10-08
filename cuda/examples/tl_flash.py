# TileLang 写的 FlashAttention 前向（因果掩码），展示在线 softmax 怎么写成 tile 级代码
from pathlib import Path

import tilelang
import tilelang.language as T


def flash_attn(batch, heads, seq, dim, block_M=64, block_N=64, num_stages=2, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504          # 乘 log2(e)，之后用 exp2 更快
    shape = (batch, seq, heads, dim)

    @T.prim_func
    def kernel(Q: T.Tensor(shape, "float16"), K: T.Tensor(shape, "float16"),
               V: T.Tensor(shape, "float16"), Out: T.Tensor(shape, "float16")):
        with T.Kernel(T.ceildiv(seq, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_M, dim), "float16")
            K_shared = T.alloc_shared((block_N, dim), "float16")
            V_shared = T.alloc_shared((block_N, dim), "float16")
            scores = T.alloc_fragment((block_M, block_N), "float")     # QK^T 的一块
            probs = T.alloc_fragment((block_M, block_N), "float16")
            acc = T.alloc_fragment((block_M, dim), "float")            # 输出累加器
            row_max = T.alloc_fragment((block_M,), "float")            # 在线 softmax 的两个统计量
            row_sum = T.alloc_fragment((block_M,), "float")
            prev_max = T.alloc_fragment((block_M,), "float")
            rescale = T.alloc_fragment((block_M,), "float")

            T.copy(Q[bz, bx * block_M, by, 0], Q_shared)
            T.fill(acc, 0)
            T.fill(row_sum, 0)
            T.fill(row_max, -T.infinity(scores.dtype))
            loop_end = T.ceildiv((bx + 1) * block_M, block_N)          # 因果掩码：只看自己之前的块
            for ko in T.Pipelined(loop_end, num_stages=num_stages):
                T.copy(K[bz, ko * block_N, by, 0], K_shared)
                T.clear(scores)
                T.gemm(Q_shared, K_shared, scores, transpose_B=True)   # QK^T
                for i, j in T.Parallel(block_M, block_N):              # 块内的因果掩码
                    scores[i, j] = T.if_then_else(bx * block_M + i >= ko * block_N + j,
                                                  scores[i, j] * scale, -T.infinity(scores.dtype))
                T.copy(row_max, prev_max)
                T.reduce_max(scores, row_max, dim=1, clear=False)      # 更新行最大值
                for i in T.Parallel(block_M):
                    rescale[i] = T.exp2(prev_max[i] - row_max[i])      # 旧的累加值要按新最大值缩放
                for i, j in T.Parallel(block_M, block_N):
                    probs[i, j] = T.exp2(scores[i, j] - row_max[i])
                for i, j in T.Parallel(block_M, dim):
                    acc[i, j] *= rescale[i]
                for i in T.Parallel(block_M):
                    row_sum[i] *= rescale[i]
                T.reduce_sum(probs, row_sum, dim=1, clear=False)
                T.copy(V[bz, ko * block_N, by, 0], V_shared)
                T.gemm(probs, V_shared, acc)                           # PV
            for i, j in T.Parallel(block_M, dim):
                acc[i, j] /= row_sum[i]                                # 最后统一除以归一化因子
            T.copy(acc, Out[bz, bx * block_M, by, 0])
    return kernel


target = {"kind": "cuda", "arch": "sm_90a"}
jit_kernel = tilelang.compile(flash_attn(1, 8, 1024, 64), out_idx=[3], target=target)
src = jit_kernel.get_kernel_source()
Path("flash.cu").write_text(src)
print(len(src.splitlines()), "行 CUDA 源码")
