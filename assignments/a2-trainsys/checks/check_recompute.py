"""torchrun --nproc-per-node 1 checks/check_recompute.py"""
import torch

from common import report, setup
from trainsys.recompute import checkpoint

setup()
torch.manual_seed(0)
blocks = [torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 256)) for _ in range(4)]
x0 = torch.randn(64, 256, requires_grad=True)


def run(use_ckpt):
    for b in blocks:
        b.zero_grad()
    x0.grad = None
    saved = [0]

    def pack(t):
        saved[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        x = x0
        for b in blocks:
            x = x + (checkpoint(b, x) if use_ckpt else b(x))
        loss = x.pow(2).mean()
    loss.backward()
    grads = [x0.grad.clone()] + [p.grad.clone() for b in blocks for p in b.parameters()]
    return saved[0], grads


plain_bytes, plain = run(False)
ckpt_bytes, ckpt = run(True)
same = all(torch.allclose(a, b, atol=1e-5) for a, b in zip(plain, ckpt))
report("checkpoint", same and ckpt_bytes <= 0.5 * plain_bytes,
       f"梯度一致：{same}；为反向保存 {ckpt_bytes / 2**20:.1f} MiB（不重计算时 {plain_bytes / 2**20:.1f} MiB）")
