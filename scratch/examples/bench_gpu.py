# 在你自己的卡上测两个数：显存带宽和 bf16 矩阵乘算力。后面所有估算都用这两个数当分母。
import time

import torch

assert torch.cuda.is_available()
dev = torch.device("cuda")
print(torch.cuda.get_device_name(0), f"| 显存 {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB",
      f"| 计算能力 sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")


def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


n = 1 << 26                                     # 256 MB 的 bf16 张量，纯访存
a = torch.randn(n, dtype=torch.bfloat16, device=dev)
b = torch.empty_like(a)
dt = timed(lambda: b.copy_(a))
print(f"显存带宽（读+写）{2 * a.numel() * 2 / dt / 1e9:.0f} GB/s")

for m in (2048, 4096, 8192):                    # 方阵矩阵乘，纯算力
    x = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
    y = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
    dt = timed(lambda: torch.mm(x, y))
    print(f"bf16 矩阵乘 {m}×{m}：{2 * m ** 3 / dt / 1e12:.0f} TFLOPS")
