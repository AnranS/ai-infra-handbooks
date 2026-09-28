from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    """还在等待（或正在分块）prefill 的请求。"""

    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None  # 分块 prefill 进行到一半时，记住已经做完的部分

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens
