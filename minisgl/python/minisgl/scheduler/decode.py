from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """一轮前向之后：把本轮的请求并入运行集合，去掉已经不能再生成的。"""
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """运行中的请求未来还可能占用多少 token 的 KV（每个请求再预留一页的余量）。"""
        reserved = (self.page_size - 1) * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        # 按 uid 排序：TP 的各个 rank 上集合的遍历顺序可能不同，排序保证大家组出同一个 batch
        return Batch(reqs=sorted(self.running_reqs, key=lambda r: r.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
