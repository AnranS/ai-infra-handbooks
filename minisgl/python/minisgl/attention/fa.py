"""FlashAttention 后端（GPU，Hopper 上用 FA3）。

flash_attn_with_kvcache 直接读分页的 KV 池：page_table 的每一行是一个请求的页号，
cache_seqlens 是每个请求的 KV 长度，cu_seqlens_q 描述本轮变长的 query。与 FlashInfer 不同，
它没有 plan 步骤，元数据就是几个张量；CUDA Graph 时把它们拷进固定缓冲区即可。
CPU 上测试时，用 tests/fakes/sgl_kernel 这个同接口的 PyTorch 实现替代。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.utils import pin

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int
    page_table: torch.Tensor  # [bs, 页数]：页号（不是 token 位置）

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.kvcache = ctx.kv_cache
        self.device = self.kvcache.device
        self.page_size = ctx.page_size
        self.scale = config.head_dim**-0.5
        self.capture: BaseCaptureData | None = None
        self.capture_bs: List[int] = []

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int,
                batch: Batch) -> torch.Tensor:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return flash_attn_with_kvcache(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            causal=True,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        bs = len(reqs)
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        max_seqlen_k, max_seqlen_q = max(seqlens_k), max(seqlens_q)
        kw = dict(dtype=torch.int32, pin_memory=pin(self.device))
        cache_seqlens = torch.tensor(seqlens_k, **kw).to(self.device, non_blocking=True)
        cu_k = torch.tensor([0] + seqlens_k, **kw).cumsum_(dim=0).to(self.device, non_blocking=True)
        if max_seqlen_q == 1:
            cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
        elif all(r.cached_len == 0 for r in reqs):
            cu_q = cu_k
        else:
            cu_q = torch.tensor([0] + seqlens_q, **kw).cumsum_(dim=0).to(self.device,
                                                                          non_blocking=True)
        # 全局 page table 按 token 存位置；每隔 page_size 取一个，再除以 page_size 就是页号
        page_table = get_global_ctx().page_table
        table = torch.stack([page_table[r.table_idx, :max_seqlen_k : self.page_size] for r in reqs])
        if self.page_size > 1:
            table = table.div(self.page_size, rounding_mode="floor")
        batch.attn_metadata = FAMetadata(cu_k, cu_q, cache_seqlens, max_seqlen_k, max_seqlen_q, table)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None
        self.capture = BaseCaptureData.create(max(bs_list), max_seq_len // self.page_size,
                                              self.device)
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and self.capture is not None
        cap = self.capture
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cap.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=cap.cu_seqlens_q[: bs + 1],
            cache_seqlens=cap.seq_lens[:bs],
            max_seqlen_k=cap.page_table.size(1) * self.page_size,
            max_seqlen_q=1,
            page_table=cap.page_table[:bs, :],
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata) and self.capture is not None
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)
