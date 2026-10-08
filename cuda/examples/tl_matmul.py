# TileLang 的 GEMM：在 Python 里直接写"块级"的搬运与矩阵乘，编译器负责线程映射与流水线
# 装：pip install tilelang；跑：需要 NVIDIA GPU（编译成 CUDA 源码则只需要 CUDA 工具链）
from pathlib import Path

import tilelang
import tilelang.language as T


def matmul(M, N, K, block_M=128, block_N=128, block_K=64, num_stages=3, threads=128):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float16"),
               B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        # 每个 block 负责输出里的一个 block_M x block_N 的 tile
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")   # 共享内存里的两块输入
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")    # 累加器放在寄存器里
            T.clear(C_local)
            # Pipelined：编译器自动把"搬下一块"和"算这一块"重叠起来，num_stages 是流水级数
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)        # 全局 -> 共享（自动用 TMA / cp.async）
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)                    # 自动展开成 mma / wgmma
            T.copy(C_local, C[by * block_M, bx * block_N])             # 累加器 -> 全局
    return kernel


target = {"kind": "cuda", "arch": "sm_90a"}       # 没有 GPU 时显式指定架构，只做代码生成
jit_kernel = tilelang.compile(matmul(1024, 1024, 1024), out_idx=[2], target=target)
src = jit_kernel.get_kernel_source()
Path("gen.cu").write_text(src)
print(len(src.splitlines()), "行 CUDA 源码")
