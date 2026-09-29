import triton
import triton.language as tl

import tritonkit


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, x_stride, y_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * x_stride + offs)          # 没有 mask：列数不是 2 的幂时会越界
    num = tl.exp(x)                                     # 没有减最大值：大输入会溢出
    tl.store(y_ptr + row * y_stride + offs, num / tl.sum(num, axis=0))


def softmax(x):
    rows, cols = x.shape
    y = tritonkit.empty_like(x)
    softmax_kernel[(rows,)](x, y, cols, cols, cols, BLOCK=triton.next_power_of_2(cols))
    return y
