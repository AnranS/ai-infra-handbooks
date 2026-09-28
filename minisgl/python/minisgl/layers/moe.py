from __future__ import annotations

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    """所有专家的权重打包成两个 3 维张量；具体怎么算交给 Context 里的 MoE 后端。

    gate_up_proj: [E, 2 * I_local, H]（每个专家的 gate 和 up 拼在一起）
    down_proj:    [E, H, I_local]
    张量并行时每个专家的中间维 I 被切开（和普通 MLP 一样），最后 all-reduce。
    """

    def __init__(self, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int,
                 renormalize: bool = True):
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        self._comm = DistributedCommunicator()
        self.tp_size = get_tp_info().size
        local = div_even(intermediate_size, self.tp_size)
        self.gate_up_proj = torch.empty(num_experts, 2 * local, hidden_size)
        self.down_proj = torch.empty(num_experts, hidden_size, local)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        out = get_global_ctx().moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._comm.all_reduce(out) if self.tp_size > 1 else out
