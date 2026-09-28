"""FlashInfer 后端（GPU）。

FlashInfer 的用法分两步：plan() 把这一批请求的"形状信息"（每条序列多长、KV 在哪些页）交给
wrapper，由它在 CPU 上算好调度方案并异步拷到 GPU；run() 对每一层执行注意力。plan 每个 batch
只做一次（在第一层前向时），run 每层都做。

我们沿用官方的做法：KV 池按 page_size = 1 展平，page table 里存的是 token 级位置，于是
paged_kv_indices 就是每个请求的 token 位置，last_page_len 恒为 1。
CPU 上没有 FlashInfer，测试时用 tests/fakes/flashinfer 这个同接口的 PyTorch 实现替代。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import create_event, div_even, pin

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << math.ceil(math.log2(n))


@dataclass
class FICaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:  # last_page_len 恒为 1，复用全 1 的 seq_lens 缓冲区
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    cu_seqlens_q_cpu: torch.Tensor
    cu_seqlens_k_cpu: torch.Tensor
    cu_seqlens_q_gpu: torch.Tensor
    indices: torch.Tensor  # 所有请求的 KV 位置拼在一起（在设备上）
    last_page_len_cpu: torch.Tensor
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    seq_lens_cpu: torch.Tensor
    dtype: torch.dtype
    wrapper: Any
    initialized: bool = False

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.float_workspace_buffer = torch.empty(128 << 20, dtype=torch.uint8, device=self.device)
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer, kv_layout="NHD", backend="fa2")
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer, use_tensor_cores=self.use_tensor_cores, kv_layout="NHD",
            backend="fa2")
        # 两个 wrapper 共用同一块整数工作区
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrapper._int_workspace_buffer = self.int_workspace_buffer
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.cached_ones_cpu = torch.tensor([], dtype=torch.int32)
        self.capture_bs: List[int] = []
        self.graph_wrappers: Dict[int, Any] = {}
        self.capture: FICaptureData | None = None
        self.last_event = create_event(self.device)
        self.last_event.record()

    @cached_property
    def use_tensor_cores(self) -> bool:
        # GQA 组越大，decode 越像小矩阵乘，用 Tensor Core 版本的 kernel 更快
        return self.config.num_qo_heads // self.config.num_kv_heads >= 4

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # plan 会复用一块锁页内存做异步拷贝，必须等上一次拷贝结束再改写它
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu, indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu, num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads, head_dim=metadata.head_dim, page_size=1,
                pos_encoding_mode="NONE", seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype, q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype, non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu, paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices, paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads, num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim, page_size=1, pos_encoding_mode="NONE",
                seq_lens=metadata.seq_lens_cpu, q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype, non_blocking=True, causal=True,
            )
        self.last_event.record()

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs > len(self.cached_ones_cpu):
            self.cached_ones_cpu = torch.ones(_next_power_of_2(bs), dtype=torch.int32,
                                              pin_memory=pin(self.device))
        return self.cached_ones_cpu[:bs]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int,
                batch: Batch) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)  # 第一层时 plan，之后各层直接 run
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        def flatten(cache: torch.Tensor) -> torch.Tensor:  # [页, 页大小, H, D] -> [token, 1, H, D]
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        kv = (flatten(self.kvcache.k_cache(layer_id)), flatten(self.kvcache.v_cache(layer_id)))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        bs = len(reqs)
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        kw = dict(dtype=torch.int32, pin_memory=pin(self.device))
        seq_lens_cpu = torch.tensor(seqlens_k, **kw)
        cu_k = torch.tensor([0] + seqlens_k, **kw).cumsum_(dim=0)
        if max(seqlens_q) == 1:  # decode：每个请求 1 个 query
            cu_q = torch.arange(0, bs + 1, **kw)
        elif all(r.cached_len == 0 for r in reqs):  # 没有命中缓存的 prefill：q 与 k 等长
            cu_q = cu_k
        else:
            cu_q = torch.tensor([0] + seqlens_q, **kw).cumsum_(dim=0)
        page_table = get_global_ctx().page_table
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_q,
            cu_seqlens_k_cpu=cu_k,
            cu_seqlens_q_gpu=cu_q.to(self.device, non_blocking=True),
            indices=torch.cat([page_table[r.table_idx, : r.device_len] for r in reqs]),
            last_page_len_cpu=self._get_ones_cpu(bs),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            seq_lens_cpu=seq_lens_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrapper if batch.is_decode else self.prefill_wrapper,
        )

    # ------------------------------------------------------------------ CUDA Graph
    # FlashInfer 为 CUDA Graph 提供了专门的 wrapper：构造时传入固定地址的缓冲区，
    # plan() 会把新的元数据拷进这些缓冲区，于是 replay 前重新 plan 一次即可。
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None
        capture = FICaptureData.create(max(bs_list), max_seq_len, self.device)
        capture.page_table = capture.page_table.view(-1)  # 当作一维的、不定长的 indices 缓冲区
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        cap = self.capture
        wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer, kv_layout="NHD", use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=cap.cu_seqlens_k[: bs + 1], indices_buffer=cap.indices,
            last_page_len_buffer=cap.one_tensor[:bs],
        )
        wrapper._backend = "fa2"
        wrapper._int_workspace_buffer = self.int_workspace_buffer
        self.graph_wrappers[bs] = wrapper
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = wrapper
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)  # 把本轮的元数据 plan 进固定缓冲区
