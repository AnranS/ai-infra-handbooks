from __future__ import annotations

import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    def __init__(self, empty: torch.Tensor):
        super().__init__(cached_len=0)
        object.__setattr__(self, "_empty", empty)

    def get_matched_indices(self) -> torch.Tensor:
        return self._empty  # type: ignore[attr-defined]


class NaivePrefixCache(BasePrefixCache):
    """不做任何前缀复用：永远不命中，插入什么都不保留（调用方随后会把页全部释放）。"""

    def __init__(self, device: torch.device):
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(NaiveCacheHandle(self.empty_tensor))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return InsertResult(0, NaiveCacheHandle(self.empty_tensor))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        raise RuntimeError("NaivePrefixCache has nothing to evict")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass
