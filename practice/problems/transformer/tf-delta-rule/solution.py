import torch


def delta_rule(q, k, v, g, beta, S=None):
    B, T, H, dk = q.shape
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / dk ** 0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if S is None:
        S = q.new_zeros(B, H, dk, v.shape[-1])
    out = torch.empty_like(v)
    for t in range(T):
        S = S * g[:, t].exp()[:, :, None, None]
        pred = torch.einsum("bhk,bhkv->bhv", k[:, t], S)
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], (v[:, t] - pred) * beta[:, t, :, None])
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
    return out, S
