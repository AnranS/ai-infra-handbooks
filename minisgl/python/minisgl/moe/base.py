from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                gating_output: torch.Tensor, topk: int, renormalize: bool) -> torch.Tensor: ...


def select_experts(gating_output: torch.Tensor, topk: int,
                   renormalize: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """路由：softmax 后取前 top-k 个专家，可选地把这 k 个权重重新归一化到和为 1。"""
    probs = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = probs.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids
