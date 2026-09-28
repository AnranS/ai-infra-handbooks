from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from minisgl.utils import Registry

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    InsertResult,
    MatchResult,
    SizeInfo,
)

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

SUPPORTED_CACHE_MANAGER = Registry[Callable[[torch.device], BasePrefixCache]]("Cache Manager")


@SUPPORTED_CACHE_MANAGER.register("naive")
def _create_naive(device: torch.device) -> BasePrefixCache:
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def _create_radix(device: torch.device) -> BasePrefixCache:
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


def create_kvcache_pool(model_config: ModelConfig, num_pages: int, page_size: int,
                        dtype: torch.dtype, device: torch.device) -> BaseKVCachePool:
    from .mha_pool import MHAKVCache

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        num_pages=num_pages,
        page_size=page_size,
        dtype=dtype,
        device=device,
    )


__all__ = [
    "BaseCacheHandle", "BaseKVCachePool", "BasePrefixCache", "InsertResult", "MatchResult",
    "SizeInfo", "SUPPORTED_CACHE_MANAGER", "create_prefix_cache", "create_kvcache_pool",
]
