from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from minisgl.utils import Registry, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

logger = init_logger(__name__)

SUPPORTED_ATTENTION_BACKENDS = Registry[Callable[["ModelConfig"], BaseAttnBackend]](
    "Attention Backend"
)


@SUPPORTED_ATTENTION_BACKENDS.register("torch")
def _create_torch(config: ModelConfig) -> BaseAttnBackend:
    from .torch_backend import TorchAttnBackend

    return TorchAttnBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register("fi")
def _create_fi(config: ModelConfig) -> BaseAttnBackend:
    from .fi import FlashInferBackend

    return FlashInferBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register("fa")
def _create_fa(config: ModelConfig) -> BaseAttnBackend:
    from .fa import FlashAttentionBackend

    return FlashAttentionBackend(config)


def validate_attn_backend(backend: str, allow_auto: bool = True) -> str:
    if backend == "auto":
        assert allow_auto, "auto is not allowed here"
    else:
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(backend.split(","))
    return backend


def create_attention_backend(backend: str, config: ModelConfig) -> BaseAttnBackend:
    """backend 可以是单个名字，也可以是 "prefill后端,decode后端" 的组合。"""
    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        p_name, d_name = backend.split(",", 1)
        if p_name != d_name:
            logger.info(f"Hybrid attention backend: prefill={p_name}, decode={d_name}")
            return HybridBackend(create_attention_backend(p_name, config),
                                 create_attention_backend(d_name, config))
        backend = p_name
    return SUPPORTED_ATTENTION_BACKENDS[backend](config)


__all__ = ["BaseAttnBackend", "BaseAttnMetadata", "HybridBackend", "SUPPORTED_ATTENTION_BACKENDS",
           "create_attention_backend", "validate_attn_backend"]
