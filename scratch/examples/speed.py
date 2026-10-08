import time

import torch

from model import GPT, GPTConfig

cfg = GPTConfig()
data = torch.load("tokens.pt")["train"].long()
B, T = 16, cfg.seq_len
i = torch.randint(0, len(data) - T - 1, (B,), generator=torch.Generator().manual_seed(0))
x = torch.stack([data[j:j + T] for j in i])
y = torch.stack([data[j + 1:j + T + 1] for j in i])


def flops_per_token(model):
    """前向 + 反向 ≈ 6 × 参数量，再加上注意力里 QKᵀ 和 PV 的 12·L·T·d（前向 2 次矩阵乘 × 2 FLOP，反向再 2 倍）"""
    return 6 * model.num_params() + 12 * cfg.n_layer * T * cfg.d_model


def measure(name, model, autocast=False, steps=10):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for s in range(steps + 3):
        if s == 3:
            t0 = time.perf_counter()                          # 前几步是预热（torch.compile 在这里编译）
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dt = (time.perf_counter() - t0) / steps
    tps = B * T / dt
    print(f"{name:22s} 每步 {dt * 1e3:6.1f} ms，{tps:8.0f} token/s，实际算力 {tps * flops_per_token(model) / 1e9:5.1f} GFLOPS")


torch.manual_seed(0)
measure("fp32", GPT(cfg))
torch.manual_seed(0)
measure("bf16 autocast", GPT(cfg), autocast=True)
torch.manual_seed(0)
t0 = time.perf_counter()
compiled = torch.compile(GPT(cfg))
measure("fp32 + torch.compile", compiled)
