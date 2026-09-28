"""sgl_kernel.flash_attn.flash_attn_with_kvcache 的"假"实现（PyTorch）。

page_table[i] 是第 i 个请求的页号，cache_seqlens[i] 是它的 KV 长度，cu_seqlens_q 划分变长的 query。
"""

from __future__ import annotations

import torch

from _fake_ref import paged_attention


def flash_attn_with_kvcache(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                            page_table: torch.Tensor, cache_seqlens: torch.Tensor,
                            cu_seqlens_q: torch.Tensor, cu_seqlens_k_new: torch.Tensor | None = None,
                            max_seqlen_q: int = 1, softmax_scale: float | None = None,
                            causal: bool = True, **kwargs) -> torch.Tensor:
    assert causal
    page_size = k_cache.shape[1]
    lens = cache_seqlens.tolist()
    pages = [page_table[i, : (n + page_size - 1) // page_size] for i, n in enumerate(lens)]
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    return paged_attention(q, k_cache, v_cache, cu_seqlens_q.tolist(), pages, lens, scale)
