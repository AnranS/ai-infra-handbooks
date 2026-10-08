"""算子分派：CUDA 张量优先用 FlashInfer（如果装了），否则用 PyTorch 参考实现。"""

from __future__ import annotations

import functools
from typing import Any, Tuple

import torch

from . import torch_ops


@functools.cache
def _flashinfer() -> Any:
    try:
        import flashinfer
    except ImportError:
        return None
    return flashinfer


# FlashInfer 的 kernel 只为半精度编译（dispatch 宏就叫 DISPATCH_..._FP16）：fp32 传进去会在
# C++ 层抛 "failed to dispatch data type"。fp32 只在与 Hugging Face 逐 token 对比时用，
# 这种场合本来就该走 PyTorch 参考实现，所以这里直接按 dtype 分派。
_FI_DTYPES = (torch.float16, torch.bfloat16)


def _use_fi(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in _FI_DTYPES and _flashinfer() is not None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None):
    if _use_fi(x):
        return _flashinfer().rmsnorm(x, weight, eps, out=out)
    return torch_ops.rmsnorm(x, weight, eps, out=out)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    if _use_fi(x):
        return _flashinfer().fused_add_rmsnorm(x, residual, weight, eps)
    return torch_ops.fused_add_rmsnorm(x, residual, weight, eps)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if _use_fi(x):
        return _flashinfer().silu_and_mul(x, out=out)
    return torch_ops.silu_and_mul(x, out=out)


def apply_rope_inplace(positions, query, key, head_size: int, cos_sin_cache) -> None:
    if _use_fi(query):
        _flashinfer().apply_rope_with_cos_sin_cache_inplace(
            positions=positions, query=query, key=key, head_size=head_size,
            cos_sin_cache=cos_sin_cache,
        )
    else:
        torch_ops.apply_rope_inplace(positions, query, key, head_size, cos_sin_cache)


def _cuda_ext(x: torch.Tensor) -> Any:
    if not x.is_cuda:
        return None
    from .cuda_ext import load_ext

    return load_ext()


def store_cache(k_cache, v_cache, indices, k, v) -> None:
    if (ext := _cuda_ext(k)) is not None:  # 第 19 章的自定义 CUDA kernel
        n = k.shape[0]
        ext.store_kv(k_cache.view(k_cache.shape[0], -1), v_cache.view(v_cache.shape[0], -1),
                     indices.int(), k.reshape(n, -1), v.reshape(n, -1),
                     torch.cuda.current_stream().cuda_stream)
        return
    torch_ops.store_cache(k_cache, v_cache, indices, k, v)


def indexing(weights: torch.Tensor, indices: torch.Tensor,
             vocab_range: Tuple[int, int] | None = None) -> torch.Tensor:
    if (ext := _cuda_ext(weights)) is not None:
        start, length = vocab_range if vocab_range is not None else (0, weights.shape[0])
        return ext.indexing(weights, indices.int(), start, length,
                            torch.cuda.current_stream().cuda_stream)
    return torch_ops.indexing(weights, indices, vocab_range)


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    return torch_ops.fast_compare_key(x, y)


__all__ = [
    "rmsnorm", "fused_add_rmsnorm", "silu_and_mul", "apply_rope_inplace",
    "store_cache", "indexing", "fast_compare_key",
]
