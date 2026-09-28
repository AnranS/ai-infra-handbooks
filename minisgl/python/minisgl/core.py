"""核心数据结构：SamplingParams、Req、Batch、Context。

整个系统里流动的就是这几样东西：调度器创建 Req、把若干 Req 组成 Batch 交给引擎；
模型的每一层不接收 batch 参数，而是从全局的 Context 里读取"当前正在计算的 batch"。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)  # eq=False：按对象身份比较和哈希，Req 可以放进 set
class Req:
    input_ids: torch.Tensor  # CPU 上的 1 维 int32 张量：提示词 + 已生成的 token
    table_idx: int  # 在 page table / token pool 中占用的行号
    cached_len: int  # 前 cached_len 个 token 的 KV 已经在缓存里
    output_len: int  # 最多还能生成多少个 token（max_tokens）
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle  # 前缀缓存中锁住的那一段

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        # device_len：本轮前向结束后，缓存里会有多少个 token 的 KV
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """本轮要送进模型的 token 数：prefill 时是未命中缓存的部分，decode 时是 1。"""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """一轮前向完成：本轮的 token 都进了缓存，下一轮再多算 1 个 token。"""
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"Req(uid={self.uid}, table_idx={self.table_idx}, cached_len={self.cached_len}, "
            f"device_len={self.device_len}, max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # 以下字段由调度器填写
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)  # 本轮新 token 的 KV 写到 KV 池的哪些位置
    padded_reqs: List[Req] = field(init=False)  # 为 CUDA Graph 补齐后的请求列表
    # 以下字段由注意力后端填写
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # 注意：page_table 总是按 page_size = 1 存放，每个元素是一个 token 在 KV 池中的位置
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch) -> Iterator[None]:
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context) -> None:
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX


def reset_global_ctx() -> None:
    """测试用：在同一个进程里反复创建引擎时，先清掉上一个全局上下文。"""
    global _GLOBAL_CTX
    _GLOBAL_CTX = None
