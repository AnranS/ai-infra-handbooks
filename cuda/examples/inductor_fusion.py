import re

import torch
from torch._inductor.utils import run_and_get_code


def rmsnorm_silu(x, w):
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    return torch.nn.functional.silu(h)


torch.manual_seed(0)
x, w = torch.randn(8, 4096), torch.randn(4096)
out, codes = run_and_get_code(torch.compile(rmsnorm_silu), x, w)   # 同时拿到结果和生成的代码
print("与 eager 结果一致：", torch.allclose(out, rmsnorm_silu(x, w), atol=1e-5))
print("生成的 kernel：", sorted(set(re.findall(r"cpp_fused_\w+", "\n".join(codes)))))
