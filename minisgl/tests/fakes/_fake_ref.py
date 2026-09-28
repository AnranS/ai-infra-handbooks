"""两个假 kernel 库共用的参考实现：分页、变长、右下角对齐的因果注意力。"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                    q_starts: List[int], kv_pages: List[torch.Tensor], kv_lens: List[int],
                    scale: float) -> torch.Tensor:
    """q: [总 query 数, Hq, D]；k_cache: [页数, 页大小, Hkv, D]。

    第 i 个请求的 query 是 q[q_starts[i]:q_starts[i+1]]，KV 是 kv_pages[i] 这些页拼起来的前 kv_lens[i]
    个 token。query 是序列的最后 n 个位置（因果掩码右下角对齐）。
    """
    out = torch.empty_like(q)
    for i, (pages, kv_len) in enumerate(zip(kv_pages, kv_lens)):
        qs, qe = q_starts[i], q_starts[i + 1]
        n = qe - qs
        if n == 0:
            continue
        k = k_cache[pages.long()].flatten(0, 1)[:kv_len].transpose(0, 1)  # [Hkv, L, D]
        v = v_cache[pages.long()].flatten(0, 1)[:kv_len].transpose(0, 1)
        q_pos = torch.arange(kv_len - n, kv_len, device=q.device).unsqueeze(1)
        mask = torch.arange(kv_len, device=q.device).unsqueeze(0) <= q_pos
        o = F.scaled_dot_product_attention(q[qs:qe].transpose(0, 1), k, v, attn_mask=mask,
                                           scale=scale, enable_gqa=True)
        out[qs:qe] = o.transpose(0, 1)
    return out
