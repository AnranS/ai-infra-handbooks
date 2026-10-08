# triton_matmul.py —— 分块矩阵乘法，tl.dot 自动使用 Tensor Core
# 在 CPU 上用解释器运行：TRITON_INTERPRET=1 python triton_matmul.py
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    # 分组排列程序实例：相邻的实例共享 A 的行块，提高 L2 命中率
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # BLOCK_M x BLOCK_K 的指针块
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn   # BLOCK_K x BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)                   # FP16/BF16 输入时走 Tensor Core，FP32 累加
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32   # CPU 解释器下用 FP32 验证逻辑
    a = torch.randn(200, 150, device=DEVICE, dtype=dtype)           # 故意取非块大小整数倍的形状
    b = torch.randn(150, 300, device=DEVICE, dtype=dtype)
    c = matmul(a, b)
    tol = 1e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(c, a @ b, rtol=tol, atol=tol)
    print("PASS triton matmul")
    if DEVICE == "cuda":
        a = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
        b = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
        ms = triton.testing.do_bench(lambda: matmul(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32))
        ref = triton.testing.do_bench(lambda: a @ b)
        flops = 2 * 4096 ** 3
        print(f"triton {flops / ms / 1e9:.1f} TFLOPS vs torch (cuBLAS) {flops / ref / 1e9:.1f} TFLOPS")
