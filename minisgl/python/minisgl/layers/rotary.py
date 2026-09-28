from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch
from minisgl.kernel import apply_rope_inplace

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(self, head_size: int, rotary_dim: int, max_position: int, base: float,
                 post_process: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        assert rotary_dim == head_size
        self.head_size = head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)  # [max_pos, D/2]
        # 以下划线开头：不是权重，不参与 state_dict；启动时一次性算好，前向时按位置查表
        self._cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    def forward(self, positions: torch.Tensor, query: torch.Tensor,
                key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        apply_rope_inplace(positions, query, key, self.head_size, self._cos_sin_cache)
        return query, key


def _llama3_post_process(scaling: Dict[str, Any]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Llama 3.1 的长上下文扩展：高频分量不变，低频分量除以 factor，中间平滑过渡。"""
    factor = scaling["factor"]
    low, high = scaling["low_freq_factor"], scaling["high_freq_factor"]
    orig_max = scaling["original_max_position_embeddings"]

    def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
        wave_len = 2 * math.pi / inv_freq
        if low == high:
            return torch.where(wave_len < orig_max / high, inv_freq, inv_freq / factor)
        smooth = ((orig_max / wave_len - low) / (high - low)).clamp(0, 1)
        return ((1 - smooth) / factor + smooth) * inv_freq

    return post_process


def _get_rope(head_dim: int, rotary_dim: int, max_position: int, base: float,
              scaling: Dict[str, Any] | None) -> RotaryEmbedding:
    rope_type = (scaling or {}).get("rope_type", "default")
    if rope_type == "default":
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    if rope_type == "llama3":
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base,
                               _llama3_post_process(scaling or {}))
    raise ValueError(f"Unsupported rope scaling: {scaling}")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device) -> None:
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache  # 所有层共享同一个 RoPE 表
def get_rope(head_dim: int, rotary_dim: int, max_position: int, base: float,
             rope_scaling: Tuple[Tuple[str, Any], ...] | None = None) -> RotaryEmbedding:
    scaling = dict(rope_scaling) if rope_scaling is not None else None
    if torch.empty(0).device.type == "meta":
        # 模型在 meta 设备上构建，但 RoPE 表是真实数据，必须放到真实设备上
        if _ROPE_DEVICE is None:
            raise RuntimeError("Call set_rope_device() before building a model on meta device")
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, scaling)
    return _get_rope(head_dim, rotary_dim, max_position, base, scaling)
