from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


class BaseKVCachePool(ABC):
    """KV 池：存放所有请求的 K、V。它只管"存"，不管"哪个位置属于谁"。"""

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor,
                 layer_id: int) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """前缀缓存匹配的结果：命中了多长，以及怎样取出这些 token 在 KV 池中的位置。"""

    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


class SizeInfo(NamedTuple):
    evictable_size: int  # 没有请求在用、可以淘汰的 token 数
    protected_size: int  # 正被请求使用（锁住）的 token 数

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    cached_len: int  # 插入前已经在缓存里的长度（调用方要释放自己那份重复的页）
    handle: BaseCacheHandle  # 插入后，指向这段前缀的句柄


class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle


class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """加锁（或解锁）一个句柄：加锁后这段前缀不会被淘汰。只改变计数，不改变缓存内容。"""

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """查找最长的已缓存前缀。返回的句柄在加锁之前随时可能被淘汰。"""

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """把一段 token 及其 KV 位置插入缓存。"""

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """淘汰至少 size 个 token，返回被释放的 KV 位置。evict(0) 总是安全的空操作。"""

    @abstractmethod
    def reset(self) -> None: ...

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo: ...

    @abstractmethod
    def check_integrity(self) -> None: ...
