from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.kernel import store_cache
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """一整块预分配的 KV 显存：[2, 层数, 页数, 页大小, 本 rank 的 KV 头数, head_dim]。"""

    def __init__(self, num_kv_heads: int, num_layers: int, head_dim: int, num_pages: int,
                 page_size: int, dtype: torch.dtype, device: torch.device) -> None:
        local_kv_heads = div_even(num_kv_heads, get_tp_info().size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device, dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._num_layers = num_layers
        self._device = torch.device(device)
        # 按 token 展平后的形状：第 i 行就是 KV 池里第 i 个 token 的位置
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor,
                 layer_id: int) -> None:
        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc, k=k, v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
