from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager


class ChunkedReq(Req):
    """只 prefill 了一部分的请求：这一轮不采样，也不进入 decode。"""

    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False


@dataclass
class PrefillAdder:
    """组一个 prefill batch 时的"记账本"：还剩多少 token 预算、已经承诺了多少 KV 空间。"""

    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # 准入控制：这个请求最坏情况下需要的 KV 空间（剩余输入 + 最大输出）必须放得下
        estimated_len = req.input_len - cached_len + req.output_len
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        # 加锁后命中的部分不再可淘汰，可用空间可能变小，需要再检查一次
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None
        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # 命中的前缀：token 和 KV 位置直接填进这一行
            pin = self.cache_manager.device.type == "cuda"
            src = req.input_ids[:cached_len].pin_memory() if pin else req.input_ids[:cached_len]
            self.table_manager.token_pool[table_idx, :cached_len].copy_(src, non_blocking=True)
            self.table_manager.page_table[table_idx, :cached_len].copy_(handle.get_matched_indices())
        return handle, table_idx

    def _add_one_req(self, pending_req: PendingReq, cache_handle: BaseCacheHandle,
                     table_idx: int, cached_len: int) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # 只把本块的 token 写进 token pool；KV 页由调度器在 _prepare_batch 里统一分配
        s = slice(cached_len, cached_len + chunk_size)
        pin = self.cache_manager.device.type == "cuda"
        src = pending_req.input_ids[s].pin_memory() if pin else pending_req.input_ids[s]
        self.table_manager.token_pool[table_idx, s].copy_(src, non_blocking=True)
        cls = ChunkedReq if is_chunked else Req
        return cls(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None
        if chunked := pending_req.chunked_req:  # 分块进行中：资源早已分好，接着做下一块
            return self._add_one_req(pending_req, chunked.cache_handle, chunked.table_idx,
                                     chunked.cached_len)
        if resource := self._try_allocate_one(pending_req):
            handle, table_idx = resource
            return self._add_one_req(pending_req, handle, table_idx, handle.cached_len)
        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if not self.pending_list:
            return None
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,  # 正在 decode 的请求已经承诺的空间
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:  # 先来先服务：遇到第一个放不下的就停
            req = adder.try_add_one(pending_req)
            if req is None:
                break
            pending_req.chunked_req = None
            if isinstance(req, ChunkedReq):
                pending_req.chunked_req = req
                chunked_list.append(pending_req)
            reqs.append(req)
        if not reqs:
            return None
        # 没做完的分块请求排到队首，下一轮优先继续
        self.pending_list = chunked_list + self.pending_list[len(reqs):]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req  # 分块中的请求已经占了资源，需要调用方释放
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
