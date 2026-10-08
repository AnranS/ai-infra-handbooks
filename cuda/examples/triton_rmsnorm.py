# triton_rmsnorm.py —— 每个程序实例处理一行的 RMSNorm
# 在 CPU 上用解释器运行：TRITON_INTERPRET=1 python triton_rmsnorm.py
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    y = x * rstd * w
    tl.store(y_ptr + row * stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rows, cols = x.shape
    y = torch.empty_like(x)
    rmsnorm_kernel[(rows,)](x, w, y, x.stride(0), cols, eps, BLOCK=triton.next_power_of_2(cols))
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 1000, device=DEVICE)
    w = torch.rand(1000, device=DEVICE) + 0.5
    ref = x / torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(rmsnorm(x, w), ref, rtol=1e-5, atol=1e-5)
    print("PASS triton rmsnorm")
