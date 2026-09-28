"""FlashInfer 的"假"实现：接口与 flashinfer 的三个 wrapper 相同，计算用 PyTorch 完成。

它把 plan/run 的语义写得明明白白：
- indptr[i]:indptr[i+1] 是第 i 个请求的页在 indices 里的范围；
- 最后一页只用了 last_page_len[i] 个 token；
- prefill 的 query 由 qo_indptr 划分，decode 每个请求 1 个 query。
CUDA Graph 版本的 wrapper 在 plan 时把元数据拷进构造时给定的缓冲区，run 只读缓冲区。
"""

from __future__ import annotations

from typing import Any, Tuple

import torch

from _fake_ref import paged_attention


class _Base:
    def __init__(self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD",
                 backend: str = "auto", use_tensor_cores: bool = False) -> None:
        assert kv_layout == "NHD"
        self._int_workspace_buffer = torch.empty(8 << 20, dtype=torch.uint8)
        self._backend = backend
        self.plan_count = 0

    def _plan(self, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor,
              last_page_len: torch.Tensor, head_dim: int) -> None:
        self.plan_count += 1
        self._qo = qo_indptr.tolist()
        kv = kv_indptr.tolist()
        self._pages = [kv_indices[kv[i]:kv[i + 1]] for i in range(len(kv) - 1)]
        self._head_dim = head_dim
        self._last = last_page_len.tolist()

    def run(self, q: torch.Tensor, paged_kv_cache: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        k_cache, v_cache = paged_kv_cache
        page_size = k_cache.shape[1]
        lens = [(len(p) - 1) * page_size + last for p, last in zip(self._pages, self._last)]
        return paged_attention(q, k_cache, v_cache, self._qo, self._pages, lens,
                               self._head_dim**-0.5)


class BatchPrefillWithPagedKVCacheWrapper(_Base):
    def plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
             num_qo_heads, num_kv_heads, head_dim_qk, page_size, causal=False, **kwargs: Any):
        assert causal
        self._plan(qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, head_dim_qk)


class BatchDecodeWithPagedKVCacheWrapper(_Base):
    def plan(self, indptr, indices, last_page_len, num_qo_heads, num_kv_heads, head_dim,
             page_size, **kwargs: Any):
        bs = len(indptr) - 1
        self._plan(torch.arange(bs + 1), indptr, indices, last_page_len, head_dim)


class CUDAGraphBatchDecodeWithPagedKVCacheWrapper(BatchDecodeWithPagedKVCacheWrapper):
    def __init__(self, float_workspace_buffer, kv_layout="NHD", use_tensor_cores=False, *,
                 indptr_buffer: torch.Tensor, indices_buffer: torch.Tensor,
                 last_page_len_buffer: torch.Tensor) -> None:
        super().__init__(float_workspace_buffer, kv_layout, use_tensor_cores=use_tensor_cores)
        self._indptr_buf, self._indices_buf, self._last_buf = (
            indptr_buffer, indices_buffer, last_page_len_buffer)

    def plan(self, indptr, indices, last_page_len, num_qo_heads, num_kv_heads, head_dim,
             page_size, **kwargs: Any):
        self.plan_count += 1
        self._indptr_buf.copy_(indptr)
        self._indices_buf[: len(indices)].copy_(indices)
        self._last_buf.copy_(last_page_len)
        self._head_dim = head_dim

    def run(self, q, paged_kv_cache):
        indptr = self._indptr_buf  # replay 时只能读这些固定缓冲区
        n = int(indptr[-1])
        self._plan(torch.arange(len(indptr)), indptr, self._indices_buf[:n], self._last_buf,
                   self._head_dim)
        self.plan_count -= 1
        return super().run(q, paged_kv_cache)
