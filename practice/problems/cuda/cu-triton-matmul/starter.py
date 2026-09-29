import triton
import triton.language as tl

import tritonkit


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pass


def matmul(a, b, BM=32, BN=32, BK=32):
    M, K = a.shape
    _, N = b.shape
    c = tritonkit.empty((M, N), like=a)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](a, b, c, M, N, K, K, 1, N, 1, N, 1, BM=BM, BN=BN, BK=BK)
    return c
