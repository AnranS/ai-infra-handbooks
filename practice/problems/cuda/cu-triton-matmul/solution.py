import triton
import triton.language as tl

import tritonkit


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a, b, BM=32, BN=32, BK=32):
    M, K = a.shape
    _, N = b.shape
    c = tritonkit.empty((M, N), like=a)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](a, b, c, M, N, K, K, 1, N, 1, N, 1, BM=BM, BN=BN, BK=BK)
    return c
