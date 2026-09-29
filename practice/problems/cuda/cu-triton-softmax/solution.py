import triton
import triton.language as tl

import tritonkit


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, x_stride, y_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(y_ptr + row * y_stride + offs, y, mask=mask)


def softmax(x):
    rows, cols = x.shape
    y = tritonkit.empty_like(x)
    softmax_kernel[(rows,)](x, y, cols, cols, cols, BLOCK=triton.next_power_of_2(cols))
    return y
