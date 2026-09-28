"""PyTorch 参考注意力后端（官方没有这个后端；SGLang 正式版里对应的是 torch_native）。

它用最直白的方式实现分页、变长、带前缀缓存的因果注意力：对 batch 中的每个请求，
按 page table 从 KV 池里取出它的全部 K、V，再调用 scaled_dot_product_attention。
速度不快，但语义清楚，CPU 上也能跑，是验证 FlashInfer / FlashAttention 后端的基准。

元数据的组织方式和 FlashAttention 后端一致：page_table 的每一行是一个请求的 KV 位置，
cache_seqlens 是每个请求的 KV 长度，cu_seqlens_q 是本轮 query 的起止位置。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx
from minisgl.utils import pin

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData, make_cu_seqlens

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class TorchAttnMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor  # [bs + 1]
    cache_seqlens: torch.Tensor  # [bs]：每个请求本轮结束后的 KV 长度（= device_len）
    page_table: torch.Tensor  # [bs, max_seqlen_k]：每个请求的 KV 在池中的位置
    max_seqlen_q: int

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TorchAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.kvcache = ctx.kv_cache
        self.device = self.kvcache.device
        self.scale = config.head_dim**-0.5
        self.capture: BaseCaptureData | None = None
        self.capture_bs: List[int] = []

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int,
                batch: Batch) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TorchAttnMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)  # 先写后读：本轮的 token 也要参与注意力
        k_pool = self.kvcache.k_cache(layer_id).flatten(0, 1)  # [num_tokens, Hkv, D]
        v_pool = self.kvcache.v_cache(layer_id).flatten(0, 1)
        cu_q = metadata.cu_seqlens_q.tolist()
        seq_lens = metadata.cache_seqlens.tolist()
        out = torch.empty_like(q)
        for i, kv_len in enumerate(seq_lens):
            qs, qe = cu_q[i], cu_q[i + 1]
            n = qe - qs  # 本轮这个请求有 n 个 query，它们是序列的最后 n 个位置
            if n == 0:
                continue
            loc = metadata.page_table[i, :kv_len].long()
            qi = q[qs:qe].transpose(0, 1)  # [Hq, n, D]
            ki = k_pool[loc].transpose(0, 1)  # [Hkv, kv_len, D]
            vi = v_pool[loc].transpose(0, 1)
            # 第 r 个 query 的绝对位置是 kv_len - n + r，它能看到位置不超过自己的所有 key
            q_pos = torch.arange(kv_len - n, kv_len, device=q.device).unsqueeze(1)
            mask = torch.arange(kv_len, device=q.device).unsqueeze(0) <= q_pos
            oi = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=self.scale,
                                                enable_gqa=True)
            out[qs:qe] = oi.transpose(0, 1)
        return out

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        page_table = get_global_ctx().page_table
        use_pin = pin(self.device)
        batch.attn_metadata = TorchAttnMetadata(
            cu_seqlens_q=make_cu_seqlens(seqlens_q, self.device, use_pin),
            cache_seqlens=torch.tensor(seqlens_k, dtype=torch.int32, pin_memory=use_pin)
            .to(self.device, non_blocking=True),
            page_table=torch.stack([page_table[r.table_idx, :max_seqlen_k] for r in reqs]),
            max_seqlen_q=max(seqlens_q),
        )

    # ------------------------------------------------------------------ CUDA Graph
    # 做法与官方 FlashAttention 后端相同：捕获时让元数据指向一组固定的缓冲区，
    # replay 前把当前 batch 的元数据拷进这些缓冲区。
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        self.capture = BaseCaptureData.create(max(bs_list), max_seq_len, self.device)
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and self.capture is not None
        cap = self.capture
        batch.attn_metadata = TorchAttnMetadata(
            cu_seqlens_q=cap.cu_seqlens_q[: bs + 1],  # decode 时恒为 [0, 1, ..., bs]
            cache_seqlens=cap.seq_lens[:bs],
            page_table=cap.page_table[:bs, :],
            max_seqlen_q=1,
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, TorchAttnMetadata) and self.capture is not None
        table_len = metadata.page_table.size(1)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)
