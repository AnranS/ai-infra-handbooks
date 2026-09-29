import math

import torch

from lm.optim import AdamW, clip_grad_norm, lr_schedule


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))


def test_adamw_matches_torch():
    a, b = _model(), _model()
    opt_a = AdamW(a.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    opt_b = torch.optim.AdamW(b.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    torch.manual_seed(1)
    for _ in range(10):
        x = torch.randn(32, 8)
        for m, opt in ((a, opt_a), (b, opt_b)):
            opt.zero_grad()
            m(x).pow(2).mean().backward()
            opt.step()
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_lr_schedule():
    assert lr_schedule(0, 1.0, 0.1, 10, 110) == 0.0
    assert math.isclose(lr_schedule(5, 1.0, 0.1, 10, 110), 0.5)
    assert math.isclose(lr_schedule(10, 1.0, 0.1, 10, 110), 1.0)
    assert math.isclose(lr_schedule(60, 1.0, 0.1, 10, 110), 0.55)
    assert math.isclose(lr_schedule(110, 1.0, 0.1, 10, 110), 0.1)
    assert math.isclose(lr_schedule(500, 1.0, 0.1, 10, 110), 0.1)


def test_clip_grad_norm():
    m = _model()
    m(torch.randn(4, 8)).sum().backward()
    ps = list(m.parameters())
    before = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in ps))
    got = clip_grad_norm(ps, 0.5)
    assert math.isclose(got, before, rel_tol=1e-5)
    after = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in ps))
    assert math.isclose(after, 0.5, rel_tol=1e-4)
    assert math.isclose(clip_grad_norm(ps, 10.0), after, rel_tol=1e-5), "没超过上限时不裁剪"
