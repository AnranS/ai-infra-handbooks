from __future__ import annotations

from typing import Callable

from minisgl.utils import Registry

from .base import BaseMoeBackend, select_experts

SUPPORTED_MOE_BACKENDS = Registry[Callable[[], BaseMoeBackend]]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("torch")
def _create_torch() -> BaseMoeBackend:
    from .torch_backend import TorchMoeBackend

    return TorchMoeBackend()


@SUPPORTED_MOE_BACKENDS.register("fused")
def _create_fused() -> BaseMoeBackend:
    from .fused import FusedMoeBackend

    return FusedMoeBackend()


def create_moe_backend(name: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[name]()


__all__ = ["BaseMoeBackend", "select_experts", "create_moe_backend", "SUPPORTED_MOE_BACKENDS"]
