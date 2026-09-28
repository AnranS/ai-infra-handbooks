from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BaseCaptureData:
    """CUDA Graph 捕获时使用的固定地址缓冲区：replay 前把当前 batch 的元数据拷进来。"""

    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )


def make_cu_seqlens(lens: list, device: torch.device, pin: bool) -> torch.Tensor:
    """[3, 5, 2] -> [0, 3, 8, 10]：变长序列拼在一起时，每条序列的起止位置。"""
    cpu = torch.tensor([0] + lens, dtype=torch.int32, pin_memory=pin).cumsum_(dim=0)
    return cpu.to(device, non_blocking=True)
