from checker import check, check_close, need_torch


def _m():
    need_torch()
    import torch

    from solution import delta_rule

    return torch, delta_rule


def _ref(torch, q, k, v, g, beta, S=None):
    """最直白的参考实现：逐 batch、逐头、逐 token 用矩阵乘写"""
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    q = q / (q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt() / dk ** 0.5
    k = k / (k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    S = torch.zeros(B, H, dk, dv) if S is None else S.clone()
    out = torch.zeros(B, T, H, dv)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                S[b, h] = S[b, h] * g[b, t, h].exp()
                kt, vt = k[b, t, h][:, None], v[b, t, h][:, None]
                pred = S[b, h].T @ kt
                S[b, h] = S[b, h] + beta[b, t, h] * kt @ (vt - pred).T
                out[b, t, h] = (S[b, h].T @ q[b, t, h][:, None])[:, 0]
    return out, S


def test_example():
    torch, delta_rule = _m()
    torch.manual_seed(0)
    B, T, H, dk, dv = 2, 5, 3, 8, 6
    q, k, v = torch.randn(B, T, H, dk), torch.randn(B, T, H, dk), torch.randn(B, T, H, dv)
    g, beta = -torch.rand(B, T, H), torch.rand(B, T, H)
    out, S = delta_rule(q, k, v, g, beta)
    check(tuple(out.shape), (B, T, H, dv), "输出形状")
    check(tuple(S.shape), (B, H, dk, dv), "记忆表形状")
    ro, rS = _ref(torch, q, k, v, g, beta)
    check_close(out, ro, rtol=1e-4, atol=1e-5, what="输出和参考实现一致")
    check_close(S, rS, rtol=1e-4, atol=1e-5, what="末状态和参考实现一致")


def test_convergence():
    torch, delta_rule = _m()
    torch.manual_seed(1)
    k = torch.randn(1, 1, 1, 16).expand(1, 6, 1, 16)
    v = torch.randn(1, 1, 1, 16).expand(1, 6, 1, 16)
    out, _ = delta_rule(k, k, v, torch.zeros(1, 6, 1), torch.full((1, 6, 1), 0.5))
    d = [(out[0, t, 0] * 4 - v[0, t, 0]).norm().item() for t in range(6)]     # q 被除以了 sqrt(16)=4，乘回去
    for t in range(1, 6):
        check_close(d[t], d[t - 1] / 2, rtol=1e-3, atol=1e-4, what=f"第 {t + 1} 次写入后距离减半")


def test_resume_from_state():
    torch, delta_rule = _m()
    torch.manual_seed(2)
    B, T, H, dk, dv = 1, 9, 2, 8, 8
    q, k, v = torch.randn(B, T, H, dk), torch.randn(B, T, H, dk), torch.randn(B, T, H, dv)
    g, beta = -torch.rand(B, T, H) * 0.5, torch.rand(B, T, H)
    full, S_full = delta_rule(q, k, v, g, beta)
    a, S1 = delta_rule(q[:, :4], k[:, :4], v[:, :4], g[:, :4], beta[:, :4])
    b, S2 = delta_rule(q[:, 4:], k[:, 4:], v[:, 4:], g[:, 4:], beta[:, 4:], S1)
    check_close(torch.cat([a, b], dim=1), full, rtol=1e-5, atol=1e-6, what="分两段递推 == 一次算完")
    check_close(S2, S_full, rtol=1e-5, atol=1e-6, what="末状态一致")


def test_decay_order():
    torch, delta_rule = _m()
    torch.manual_seed(3)
    q = k = torch.randn(1, 2, 1, 4)
    v = torch.randn(1, 2, 1, 4)
    g = torch.tensor([[[0.0], [-1.0]]])
    beta = torch.ones(1, 2, 1)
    _, S = delta_rule(q, k, v, g, beta)
    _, rS = _ref(torch, q, k, v, g, beta)
    check_close(S, rS, rtol=1e-5, atol=1e-6, what="衰减在写入之前")
