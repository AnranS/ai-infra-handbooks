import torch
from muon import Muon


def state_bytes(opt):
    return sum(t.numel() * t.element_size() for s in opt.state.values() for t in s.values()
               if torch.is_tensor(t) and t.dim() > 0)          # 不算 step 这类标量


for name in ("AdamW", "Muon"):
    torch.manual_seed(0)
    w1, w2 = torch.randn(4096, 1024, requires_grad=True), torch.randn(1024, 4096, requires_grad=True)
    opt = torch.optim.AdamW([w1, w2], lr=1e-3) if name == "AdamW" else Muon([w1, w2], lr=1e-3)
    (torch.randn(8, 1024) @ w1.T @ w2.T).sum().backward()
    opt.step()
    n = w1.numel() + w2.numel()
    print(f"{name:5s}：{n / 1e6:.1f}M 个参数，优化器状态 {state_bytes(opt) / n:.0f} 字节/参数")
