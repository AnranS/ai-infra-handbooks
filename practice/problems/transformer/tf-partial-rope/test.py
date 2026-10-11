from checker import check, check_close, need_torch


def _m():
    need_torch()
    import torch

    from solution import apply_rope, rope_tables

    return torch, rope_tables, apply_rope


def test_example():
    torch, rope_tables, apply_rope = _m()
    cos, sin = rope_tables(torch.arange(6), 64, 1e7)
    check(tuple(cos.shape), (6, 64), "cos 表的形状")
    torch.manual_seed(0)
    x = torch.randn(1, 6, 8, 256)
    y = apply_rope(x, cos, sin)
    check(tuple(y.shape), (1, 6, 8, 256), "输出形状")
    check(bool(y[..., 64:].equal(x[..., 64:])), True, "后 192 维原样不动")


def test_tables():
    torch, rope_tables, _ = _m()
    cos, sin = rope_tables(torch.arange(5), 8, 10000.0)
    check_close(cos[0], torch.ones(8), rtol=0, atol=1e-6, what="位置 0 的 cos 全是 1")
    check_close(sin[0], torch.zeros(8), rtol=0, atol=1e-6, what="位置 0 的 sin 全是 0")
    check_close(cos[:, :4], cos[:, 4:], rtol=0, atol=1e-6, what="后一半重复前一半")
    inv = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
    check_close(cos[3, :4], (3 * inv).cos(), rtol=1e-6, atol=1e-6, what="角度 = 位置 × 频率")


def test_relative_position():
    torch, rope_tables, apply_rope = _m()
    torch.manual_seed(1)
    q, k = torch.randn(1, 1, 2, 32), torch.randn(1, 1, 2, 32)

    def dot(pq, pk):
        cq, sq = rope_tables(torch.tensor([pq]), 16, 1e4)
        ck, sk = rope_tables(torch.tensor([pk]), 16, 1e4)
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum(-1)

    check_close(dot(3, 1), dot(103, 101), rtol=1e-4, atol=1e-5, what="距离相同则点积相同（相对性）")
    diff = (dot(3, 1) - dot(3, 2)).abs().max().item()
    check(diff > 1e-3, True, "距离不同时点积不同")


def test_rotation_formula():
    torch, rope_tables, apply_rope = _m()
    torch.manual_seed(2)
    x = torch.randn(2, 4, 3, 8)
    cos, sin = rope_tables(torch.arange(4), 4, 100.0)
    y = apply_rope(x, cos, sin)
    a, b = x[..., 0:2], x[..., 2:4]
    c, s = cos[None, :, None, :2], sin[None, :, None, :2]
    check_close(y[..., 0:2], a * c - b * s, rtol=1e-5, atol=1e-6, what="前半：a cos - b sin")
    check_close(y[..., 2:4], b * c + a * s, rtol=1e-5, atol=1e-6, what="后半：b cos + a sin")
    check(bool(y[..., 4:].equal(x[..., 4:])), True, "没旋转的维度不变")
