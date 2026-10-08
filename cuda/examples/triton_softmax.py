# triton_softmax.py —— 每个程序实例处理一行的融合 softmax
# 在 CPU 上用解释器运行：TRITON_INTERPRET=1 python triton_softmax.py
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                    # 减去最大值，数值稳定
    num = tl.exp(x)                              # 被掩码的位置 exp(-inf) = 0
    out = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)         # 一行必须能放进一个块
    num_warps = 4 if BLOCK <= 2048 else 8 if BLOCK <= 8192 else 16
    softmax_kernel[(rows,)](out, x, x.stride(0), out.stride(0), cols, BLOCK=BLOCK, num_warps=num_warps)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(123, 781, device=DEVICE) * 10
    out = softmax(x)
    torch.testing.assert_close(out, torch.softmax(x, dim=1), rtol=1e-5, atol=1e-6)
    print("PASS triton softmax")
    if DEVICE == "cuda":
        big = torch.randn(4096, 4096, device=DEVICE)
        ms = triton.testing.do_bench(lambda: softmax(big))
        print(f"softmax 4096x4096: {ms:.3f} ms, {2 * big.numel() * 4 / ms / 1e6:.1f} GB/s")
