from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil, pin

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    """KV 池的分配器 + 前缀缓存的门面。

    free_slots 里存的是每个空闲页第一个 token 的位置（按页对齐），例如 page_size = 2 时是
    [0, 2, 4, ...]。一个页要么空闲（在 free_slots 里），要么在前缀缓存里，要么被某个请求独占。
    """

    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def match_req(self, req: PendingReq) -> MatchResult:
        # 最后一个 token 不参与匹配：即使整个提示词都命中，也至少要算一个 token 才能得到 logits
        assert req.input_len > 0
        return self.prefix_cache.match_prefix(req.input_ids[: req.input_len - 1])

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        """为本轮要计算的 token 分配 KV 页，并写进 page table。已经分过页的部分不重复分配。"""
        needed_pages = 0
        info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        """把请求已经算好的 KV 交给前缀缓存。

        [0, old.cached_len)              prefill 之前就在缓存里（本请求锁着它）
        [old.cached_len, cached_len)     本请求自己算的，但期间别的请求也把同样的前缀插进了缓存
                                         —— 缓存里已有一份，本请求这份是重复的，释放
        [cached_len, new.cached_len)     本次新插入缓存的部分，所有权转交给缓存
        [new.cached_len, req.cached_len) 不足一页、插不进缓存的尾巴：请求结束就释放，否则留着
        """
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        self.unlock(old_handle)  # 所有依赖旧句柄的操作都做完了才能解锁
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:
            self._free(page_indices[new_handle.cached_len :])
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                f"CacheManager integrity check failed: free_pages({len(self.free_slots)}) + "
                f"cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self) -> Iterator[None]:
        """在这个区域内的释放先攒着，最后一次性拼进 free_slots（少做很多次 torch.cat）。"""
        lazy_list: List[torch.Tensor] = []

        def lazy_free(indices: torch.Tensor) -> None:
            if len(indices) > 0:
                lazy_list.append(indices[:: self.page_size])

        try:
            self._free = lazy_free  # type: ignore[method-assign]
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):  # 空闲页不够：从前缀缓存里淘汰
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, X * page_size + 1, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(page_table: torch.Tensor, allocated: torch.Tensor,
                      info: List[Tuple[int, int, int]], page_size: int) -> None:
    """一次性把所有新分配的位置写进 page table：先在 CPU 上算好 (行, 列) 下标，再一次索引赋值。"""
    n = len(allocated)
    use_pin = pin(page_table.device)
    rows = torch.empty(n, dtype=torch.int64, pin_memory=use_pin)
    cols = torch.empty(n, dtype=torch.int64, pin_memory=use_pin)
    offset = 0
    for table_idx, first_page, last_page in info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        rows[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=cols[offset : offset + length])
        offset += length
    assert offset == n
    device = page_table.device
    page_table[rows.to(device, non_blocking=True), cols.to(device, non_blocking=True)] = allocated
