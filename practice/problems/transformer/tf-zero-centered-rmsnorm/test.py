from checker import check, check_close, need_torch


def _m():
    need_torch()
    import torch

    from solution import gated_rmsnorm, rmsnorm

    return torch, rmsnorm, gated_rmsnorm


def test_example():
    torch, rmsnorm, _ = _m()
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8) * 5
    y = rmsnorm(x, torch.zeros(8))
    check_close(y.pow(2).mean(-1).sqrt(), torch.ones(2, 3), rtol=1e-4, atol=1e-5, what="权重为 0 时均方根为 1")
    check(rmsnorm(x.bfloat16(), torch.zeros(8)).dtype, torch.bfloat16, "输出保持输入的 dtype")


def test_zero_centered_weight():
    torch, rmsnorm, _ = _m()
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    w = torch.randn(16) * 0.3
    ref = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * (1 + w)
    check_close(rmsnorm(x, w), ref, rtol=1e-5, atol=1e-6, what="乘的是 1 + weight")
    # 零中心的意义：bf16 存 1 + 1e-3 会被吞成 1.0，存 1e-3 不会
    tiny = torch.full((16,), 1e-3).bfloat16()
    y = rmsnorm(x, tiny.float())
    ref = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * (1 + tiny.float())
    check_close(y, ref, rtol=1e-5, atol=1e-6, what="很小的权重更新没有被吞掉")


def test_fp32_statistics():
    torch, rmsnorm, _ = _m()
    torch.manual_seed(2)
    x = (torch.randn(8, 1024) * 3).bfloat16()
    y = rmsnorm(x, torch.zeros(1024))
    ref = (x.float() / x.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()).bfloat16()
    check(y.dtype, torch.bfloat16, "bf16 输入给 bf16 输出")
    check_close(y.float(), ref.float(), rtol=1e-2, atol=1e-2, what="统计量在 fp32 里算（bf16 输入）")


def test_gated():
    torch, _, gated_rmsnorm = _m()
    import torch.nn.functional as F
    torch.manual_seed(3)
    x, g, w = torch.randn(5, 7, 32), torch.randn(5, 7, 32), torch.rand(32) + 0.5
    ref = w * (x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()) * F.silu(g)
    check_close(gated_rmsnorm(x, g, w), ref, rtol=1e-5, atol=1e-6, what="归一化后乘 weight 和 silu(门)")
    check(gated_rmsnorm(x.bfloat16(), g.bfloat16(), w).dtype, torch.bfloat16, "带门版也保持 dtype")
