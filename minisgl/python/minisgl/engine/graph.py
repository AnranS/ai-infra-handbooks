"""CUDA Graph：把 decode 的一次前向录制下来，之后用一次 replay 代替成百上千次 kernel 启动。

录制下来的 graph 里，每个 kernel 读写的显存地址都是固定的。所以：
1. 输入（input_ids、positions、out_loc）必须放在固定的缓冲区里，replay 前把新数据拷进去；
2. 注意力元数据同理，由注意力后端的 prepare_for_capture / prepare_for_replay 负责；
3. 每个批大小录一个 graph，实际的 batch 向上补齐（padding）到最近的已录制大小。

CPU 上没有 CUDA Graph。为了在 CPU 上也能验证上面这套"数据全部走固定缓冲区"的约定，
EmulatedGraph 在"捕获"时只记下要执行的函数和当时的 batch，replay 时用捕获时的 batch
（它的输入全都指向固定缓冲区）重新执行一遍前向。如果哪个输入忘了拷进缓冲区，结果就会出错。
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.utils import get_free_memory, init_logger, is_cuda, synchronize

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        """捕获时：让 batch 的输入字段直接指向缓冲区。"""
        s = slice(batch.padded_size)
        batch.input_ids, batch.out_loc, batch.positions = (
            self.input_ids[s], self.out_loc[s], self.positions[s])

    def copy_from(self, batch: Batch) -> None:
        """replay 前：把当前 batch 的输入拷进缓冲区。"""
        s = slice(batch.padded_size)
        self.input_ids[s] = batch.input_ids
        self.out_loc[s] = batch.out_loc
        self.positions[s] = batch.positions


class EmulatedGraph:
    """CPU 上的替身：接口与 torch.cuda.CUDAGraph 的 replay 一致。"""

    def __init__(self, fn: Callable[[], None], batch: Batch) -> None:
        self.fn, self.batch = fn, batch

    def replay(self) -> None:
        ctx = get_global_ctx()
        saved, ctx._batch = ctx._batch, self.batch  # 真正的 graph 不读 Python 对象，这里模拟这一点
        try:
            self.fn()
        finally:
            ctx._batch = saved


def determine_cuda_graph_bs(cuda_graph_bs: List[int] | None, cuda_graph_max_bs: int | None,
                            free_memory: int, device: torch.device) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs
    if cuda_graph_max_bs is None:
        if not is_cuda(device):
            return []  # CPU 上默认关闭（需要验证时显式传入 cuda_graph_max_bs）
        cuda_graph_max_bs = 256 if free_memory > (80 << 30) else 160
    if cuda_graph_max_bs < 1:
        return []
    return sorted({b for b in [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
                   if b <= cuda_graph_max_bs})


class GraphRunner:
    def __init__(self, stream, device: torch.device, model: BaseLLMModel,
                 attn_backend: BaseAttnBackend, cuda_graph_bs: List[int] | None,
                 cuda_graph_max_bs: int | None, free_memory: int, max_seq_len: int,
                 vocab_size: int, dummy_req: Req) -> None:
        bs_list = determine_cuda_graph_bs(cuda_graph_bs, cuda_graph_max_bs, free_memory, device)
        self.attn_backend = attn_backend
        self.graph_bs_list = sorted(bs_list)
        self.max_graph_bs = max(bs_list) if bs_list else 0
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self.graph_map: Dict[int, object] = {}
        if self.max_graph_bs > 0:
            self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel) -> None:
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)
        synchronize(self.device)
        logger.info_rank0(f"Capturing graphs for batch sizes {self.graph_bs_list}")
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        pool = None
        for bs in sorted(self.graph_bs_list, reverse=True):  # 先录最大的，内存池给小的复用
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            logits = self.buffer.logits[:bs]
            ctx = get_global_ctx()
            with ctx.forward_batch(batch):
                logits.copy_(model.forward())  # 预热一次：触发各种懒初始化，避免录进 graph
                if is_cuda(self.device):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                        logits.copy_(model.forward())
                    if pool is None:
                        pool = graph.pool()
                else:
                    graph = EmulatedGraph(lambda m=model, lg=logits: lg.copy_(m.forward()), batch)
            self.graph_map[bs] = graph
        logger.info_rank0(f"Free memory after capture: {get_free_memory(self.device) / 2**30:.2f} GiB")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def pad_batch(self, batch: Batch) -> None:
        """把 batch 补齐到最近的已录制批大小；补上的是 dummy 请求，结果丢弃。"""
        padded = (next(bs for bs in self.graph_bs_list if bs >= batch.size)
                  if self.can_use_cuda_graph(batch) else batch.size)
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded - batch.size)

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        self.attn_backend.prepare_for_replay(batch)
        self.graph_map[batch.padded_size].replay()  # type: ignore[attr-defined]
        return self.buffer.logits[: batch.size]

    def destroy_cuda_graphs(self) -> None:
        self.graph_map.clear()
        gc.collect()
