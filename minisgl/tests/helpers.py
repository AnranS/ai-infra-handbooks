"""不经过调度器、手工驱动引擎的小工具：第 4～6 章还没有调度器，用它来组 batch。"""

from __future__ import annotations

from typing import List

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig


def build_engine(path: str, **kwargs) -> Engine:
    kwargs.setdefault("dtype", torch.float32)
    kwargs.setdefault("max_running_req", 8)
    kwargs.setdefault("num_page_override", 4096)
    kwargs.setdefault("max_seq_len_override", 512)
    return Engine(EngineConfig(model_path=path, tp_info=DistributedInfo(0, 1), **kwargs))


def new_req(ids: List[int], table_idx: int, max_tokens: int = 16, cached_len: int = 0) -> Req:
    return Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx,
               cached_len=cached_len, output_len=max_tokens, uid=table_idx,
               sampling_params=SamplingParams(max_tokens=max_tokens), cache_handle=None)


def forward(engine: Engine, reqs: List[Req], phase: str) -> torch.Tensor:
    """按调度器的约定填好 batch 的各个字段，跑一次前向，返回 logits（不采样、不推进请求状态）。

    这里假设 page table 已经由调用方写好（最简单的做法：第 i 行用 [i*512, i*512+512) 这段位置）。
    """
    batch = Batch(reqs=reqs, phase=phase)  # type: ignore[arg-type]
    engine.graph_runner.pad_batch(batch)
    batch.positions = torch.cat([torch.arange(r.cached_len, r.device_len, dtype=torch.int32)
                                 for r in batch.padded_reqs])
    batch.input_ids = torch.cat([r.input_ids[r.cached_len:r.device_len] for r in batch.padded_reqs])
    batch.out_loc = torch.cat([engine.page_table[r.table_idx, r.cached_len:r.device_len]
                               for r in batch.padded_reqs])
    engine.attn_backend.prepare_metadata(batch)
    with engine.ctx.forward_batch(batch):
        return engine.model.forward()


def identity_page_table(engine: Engine, rows: int, width: int = 512) -> None:
    for i in range(rows):
        engine.page_table[i, :width] = torch.arange(i * width, (i + 1) * width, dtype=torch.int32)


def build_llm(path: str, **kwargs):
    from minisgl.llm import LLM

    kwargs.setdefault("dtype", torch.float32)
    kwargs.setdefault("max_running_req", 8)
    kwargs.setdefault("num_page_override", 1024)
    kwargs.setdefault("max_seq_len_override", 512)
    return LLM(path, **kwargs)


def greedy(max_tokens: int) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens, ignore_eos=True)
