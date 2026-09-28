"""线性层。张量并行下有四种切法，第 16 章详细讲；TP=1 时它们都退化成普通的 F.linear。"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    def __init__(self, full_isize: int, full_osize: int, local_isize: int, local_osize: int,
                 has_bias: bool):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """每个 rank 都保存完整权重（例如 MoE 的路由层）。"""

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        super().__init__(input_size, output_size, input_size, output_size, has_bias)


class LinearColParallelMerged(_LinearTPImpl):
    """按输出维切分的列并行，并把几个输入相同的投影合并成一次矩阵乘（gate_proj + up_proj）。"""

    def __init__(self, input_size: int, output_sizes: List[int], has_bias: bool):
        tp_size = get_tp_info().size
        local_sizes = [div_even(size, tp_size) for size in output_sizes]
        super().__init__(input_size, sum(output_sizes), input_size, sum(local_sizes), has_bias)


class LinearQKVMerged(_LinearTPImpl):
    """合并的 QKV 投影，按注意力头切分；KV 头数少于 TP 数时，KV 头在多个 rank 上复制。"""

    def __init__(self, hidden_size: int, head_dim: int, num_qo_heads: int, num_kv_heads: int,
                 has_bias: bool):
        tp_size = get_tp_info().size
        local_qo = div_even(num_qo_heads, tp_size)
        local_kv = div_even(num_kv_heads, tp_size, allow_replicate=True)
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_osize = (local_qo + 2 * local_kv) * head_dim
        super().__init__(hidden_size, full_osize, hidden_size, local_osize, has_bias)


class LinearRowParallel(_LinearTPImpl):
    """按输入维切分的行并行：每个 rank 算出部分和，再 all-reduce。"""

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        local_isize = div_even(input_size, tp_info.size)
        super().__init__(input_size, output_size, local_isize, output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        return self._comm.all_reduce(y) if self._tp_size > 1 else y


class LinearOProj(LinearRowParallel):
    """注意力的输出投影 o_proj，就是一个行并行的线性层。"""
