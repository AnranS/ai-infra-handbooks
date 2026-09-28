from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """prefill 时每个请求最后一个 token 在展平后的 q 中的下标（LM head 只算这些位置）。"""


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int,
                batch: Batch) -> torch.Tensor:
        """先把本轮的 k、v 写进 KV 池，再计算注意力。q: [T, Hq, D]，返回 [T, Hq, D]。"""

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """每个 batch 调用一次（而不是每层一次）：把请求信息整理成 kernel 需要的格式。"""

    # 以下三个方法服务于 CUDA Graph（第 18 章）
    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class HybridBackend(BaseAttnBackend):
    """prefill 和 decode 用不同的后端，例如 Hopper 上 prefill 用 FA3、decode 用 FlashInfer。"""

    def __init__(self, prefill_backend: BaseAttnBackend, decode_backend: BaseAttnBackend) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def _pick(self, batch: Batch) -> BaseAttnBackend:
        return self.prefill_backend if batch.is_prefill else self.decode_backend

    def forward(self, q, k, v, layer_id: int, batch: Batch):
        return self._pick(batch).forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        self._pick(batch).prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)  # 只有 decode 会被捕获

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
