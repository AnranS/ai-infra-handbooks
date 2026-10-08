# triton_add.py —— Triton 的第一个 kernel：向量加法
# 在 CPU 上用解释器运行：TRITON_INTERPRET=1 python triton_add.py
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                  # 第几个程序实例，相当于 blockIdx.x
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # 这个实例负责的 BLOCK 个下标（一个向量）
    mask = offsets < n                           # 越界的位置不读不写
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)               # 启动多少个程序实例
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(98_765, device=DEVICE)
    y = torch.randn(98_765, device=DEVICE)
    out = add(x, y)
    assert torch.allclose(out, x + y), "mismatch"
    print("PASS triton add")
