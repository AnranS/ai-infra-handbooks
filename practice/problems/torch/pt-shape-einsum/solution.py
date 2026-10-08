import torch


def attn_scores(q, k):
    d = q.shape[-1]
    return torch.einsum("bhtd,bhsd->bhts", q, k) * d**-0.5


def merge_heads(x):
    b, h, t, d = x.shape
    return x.transpose(1, 2).reshape(b, t, h * d)


def split_heads(x, h):
    b, t, hd = x.shape
    return x.view(b, t, h, hd // h).transpose(1, 2)


def broadcast_shape(sa, sb):
    out = []
    for i in range(max(len(sa), len(sb))):
        a = sa[-1 - i] if i < len(sa) else 1
        b = sb[-1 - i] if i < len(sb) else 1
        if a != b and a != 1 and b != 1:
            raise ValueError(f"形状 {tuple(sa)} 和 {tuple(sb)} 不能广播：第 {i} 维（从右数）是 {a} 和 {b}")
        out.append(max(a, b))
    return tuple(reversed(out))
