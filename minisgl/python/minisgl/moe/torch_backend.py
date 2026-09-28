from __future__ import annotations

import torch
from minisgl.kernel import silu_and_mul

from .base import BaseMoeBackend, select_experts


class TorchMoeBackend(BaseMoeBackend):
    """参考实现：按专家分组，每个专家只处理路由到它的那些 token。"""

    def forward(self, hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                gating_output: torch.Tensor, topk: int, renormalize: bool) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(gating_output, topk, renormalize)
        out = torch.zeros_like(hidden_states)
        flat_ids = topk_ids.flatten()                       # [T * k]
        token_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        token_idx = token_idx.repeat_interleave(topk)       # 第 i 个元素属于哪个 token
        flat_weights = topk_weights.flatten().to(hidden_states.dtype)
        for e in flat_ids.unique().tolist():
            sel = (flat_ids == e).nonzero().squeeze(1)
            tokens = token_idx[sel]
            h = silu_and_mul(hidden_states[tokens] @ w1[e].t())
            y = (h @ w2[e].t()) * flat_weights[sel].unsqueeze(1)
            out.index_add_(0, tokens, y)
        return out
