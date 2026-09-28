"""各个算子的 PyTorch 参考实现。

它们在 CPU 和 GPU 上都能跑，是本教程验证正确性的基准；GPU 上可以换成 FlashInfer 或
自定义 CUDA kernel（见 kernel/__init__.py 的分派逻辑）。数值上刻意与 Hugging Face 的实现
保持相同的计算顺序，这样在 float32 下可以与官方模型逐 token 对齐。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None
) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = weight * y.to(x.dtype)
    if out is not None:
        out.copy_(y)
        return out
    return y


def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """原地计算：residual <- x + residual；x <- rmsnorm(residual)。"""
    residual.add_(x)
    rmsnorm(residual, weight, eps, out=x)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """输入是 [.., 2d]，前一半是 gate、后一半是 up：返回 silu(gate) * up。"""
    d = x.shape[-1] // 2
    y = F.silu(x[..., :d]) * x[..., d:]
    if out is not None:
        out.copy_(y)
        return out
    return y


def apply_rope_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> None:
    """NeoX 风格（前后两半配对）的旋转位置编码，原地修改 query 和 key。

    query: [T, Hq * D] 或 [T, Hq, D]；key 同理；cos_sin_cache: [max_pos, D]，前一半是 cos、后一半是 sin。
    """
    cos, sin = cos_sin_cache[positions.long()].chunk(2, dim=-1)  # [T, D/2]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    for t in (query, key):
        x = t.view(t.shape[0], -1, head_size)
        x1, x2 = x[..., : head_size // 2], x[..., head_size // 2 :]
        c, s = cos.to(x.dtype), sin.to(x.dtype)
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
        x.copy_(torch.cat([o1, o2], dim=-1))


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """把本轮新算出的 K、V 按 indices 写进 KV 池。k_cache: [num_tokens, H, D]；k: [n, H*D] 或 [n, H, D]。"""
    idx = indices.long()
    k_cache[idx] = k.view(k.shape[0], *k_cache.shape[1:])
    v_cache[idx] = v.view(v.shape[0], *v_cache.shape[1:])


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    vocab_range: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """词表查找。vocab_range=(start, length) 时只查本 rank 负责的那一段，其余位置填 0。"""
    if vocab_range is None:
        return weights[indices.long()]
    start, length = vocab_range
    local = indices.long() - start
    mask = (local >= 0) & (local < length)
    out = weights[local.clamp(0, length - 1)]
    return out * mask.unsqueeze(-1).to(out.dtype)


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """返回两个 1 维整数张量从头开始相同的元素个数（第一个不同元素的下标）。"""
    n = min(len(x), len(y))
    diff = (x[:n] != y[:n]).nonzero()
    return int(diff[0, 0]) if len(diff) > 0 else n
