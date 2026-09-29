"""Triton 模拟器：向量加法"""
import numpy as np
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n                                  # 最后一个块可能越界，用掩码挡住
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


n = 1000
x, y = np.random.rand(n).astype(np.float32), np.random.rand(n).astype(np.float32)
out = np.empty_like(x)
add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
print("结果正确：", np.allclose(out, x + y), "；程序实例数：", triton.cdiv(n, 256))
