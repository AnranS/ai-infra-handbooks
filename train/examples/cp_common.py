import torch


def attention(q, k, v, causal=True, q_offset=0, k_offset=0):
    """q: [Sq, H, D], k/v: [Sk, H, D]；返回输出和每行的 log-sum-exp（用于合并分块结果）。offset 是这一块在完整序列中的起点。"""
    scores = torch.einsum("qhd,khd->hqk", q, k) / q.shape[-1] ** 0.5
    if causal:
        qi = torch.arange(q.shape[0])[:, None] + q_offset
        ki = torch.arange(k.shape[0])[None, :] + k_offset
        scores = scores.masked_fill(ki > qi, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)                     # [H, Sq]
    out = torch.einsum("hqk,khd->qhd", torch.exp(scores - lse[..., None]), v)
    return out, lse


def make_qkv(S=32, H=8, D=16):
    torch.manual_seed(0)
    return torch.randn(S, H, D), torch.randn(S, H, D), torch.randn(S, H, D)
