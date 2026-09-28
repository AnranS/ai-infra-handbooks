from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import create_stream, init_logger, load_tokenizer, pin, set_stream, stream_ctx
from minisgl.utils import synchronize

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput

logger = init_logger(__name__)

Indice2D = Tuple[torch.Tensor, torch.Tensor]


class ForwardInput(NamedTuple):
    """重叠调度时，一个 batch 的输入要一直活到它的结果被处理完，所以打包保存。"""

    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (行号, 列号)：从 token_pool 取出本轮输入
    write_tuple: Indice2D  # (行号, 列号或 -1)：把采样结果写回 token_pool


ForwardData = Tuple[ForwardInput, "ForwardOutput"]


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)
        self.config = self.engine.config
        # 调度器自己用一条流准备元数据，引擎在另一条流上计算，两者可以重叠
        self.device = self.engine.device
        self.stream = create_stream(self.device)
        self.engine_stream_ctx = stream_ctx(self.engine.stream)
        set_stream(self.stream)

        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type)
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager)

        self.finished_reqs: Set[Req] = set()
        # 重叠调度下，释放的请求槽要等"可能用到它的 batch"算完才能重新分配（见 _free_req_resources）
        self._defer_slot_free = False
        self._deferred_slots: List[int] = []
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    # ------------------------------------------------------------------ 两种主循环
    def normal_loop(self) -> None:
        """朴素循环：收消息 → 调度 → 前向 → 等结果 → 处理结果。CPU 和 GPU 轮流干活。"""
        self._defer_slot_free = False  # 普通循环里没有在途的 batch，释放立即生效
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)
        forward_input = self._schedule_next_batch()
        ongoing = None
        if forward_input is not None:
            ongoing = (forward_input, self._forward(forward_input))
        self._process_last_data(ongoing)

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """重叠循环：先发射第 N 轮的计算，再在 CPU 上处理第 N-1 轮的结果。

        GPU 执行第 N 轮的同时，CPU 在做第 N-1 轮的结果处理和第 N+1 轮的调度，CPU 开销被藏起来。
        能这样做的前提是：第 N 轮的输入 token 不需要 CPU 知道——它由第 N-1 轮的采样结果在 GPU 上
        直接写进 token_pool（见 _forward）。
        """
        self._defer_slot_free = True
        if last_data is None:  # 没有在途的 batch：之前推迟释放的请求槽可以放心复用了
            self._release_deferred_slots()
        blocking = not (last_data is not None or self.prefill_manager.runnable
                        or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)
        forward_input = self._schedule_next_batch()
        ongoing = None
        if forward_input is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)  # 等本轮的元数据准备好
                ongoing = (forward_input, self._forward(forward_input))
        self._process_last_data(last_data)
        return ongoing

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    # ------------------------------------------------------------------ 处理结果
    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()  # 等采样结果拷回 CPU；这也意味着这个 batch 以及之前的 batch 都算完了
        self._release_deferred_slots()
        reply: List[DetokenizeMsg] = []
        new_finished: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):  # 分块中的请求这一轮没有输出
                    continue
                if req in self.finished_reqs:
                    # 重叠调度下，请求在上一轮已经结束（遇到 EOS），但结束的消息还没处理，这一轮就已经
                    # 把它带上了。这一轮的结果是多余的，直接丢弃（官方实现会多发一条消息，前端恰好会忽略）。
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                token = int(next_token.item())
                # 判断是否达到 max_tokens 不能用 req.can_decode：重叠调度下，处理第 N 轮结果时
                # 第 N+1 轮已经发射，req 的 device_len 已经被推进了一步，会提前一个 token 判定结束。
                # 用 CPU 侧已经收到的 token 数判断，与调度节奏无关。
                finished = len(req.input_ids) >= req.max_device_len
                if not req.sampling_params.ignore_eos:
                    finished |= token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=token, finished=finished))
                if finished:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished.add(req)
                elif batch.is_prefill:  # prefill 刚结束：立刻把提示词的 KV 交给前缀缓存
                    self.cache_manager.cache_req(req, finished=False)
        self.finished_reqs = new_finished
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for m in msg.data:
                self._process_one_msg(m)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(f"Input length {input_len} exceeds {max_seq_len}, "
                                     f"request {msg.uid} dropped.")
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            req = self.prefill_manager.abort_req(msg.uid) or self.decode_manager.abort_req(msg.uid)
            if req is not None:
                self._free_req_resources(req)
                # 重叠调度下它可能还在在途的 batch 里：记为已结束，那一轮的结果会被丢弃，
                # 不会再对一个已经释放的请求调用 cache_req
                self.finished_reqs.add(req)
        else:
            raise NotImplementedError(f"Unknown message type: {type(msg)}")

    def _free_req_resources(self, req: Req) -> None:
        # 请求槽（page table / token pool 的一行）：重叠调度下，GPU 上可能还有一个包含这个请求的
        # batch 在跑（它会往 token_pool 的这一行写采样结果）。如果此刻把这一行分给新请求，调度器流上
        # 对这一行的写入会与引擎流上的旧 batch 竞争。所以推迟到下一次处理结果、确认在途 batch 算完之后。
        # KV 页不需要推迟：新请求写 KV 的 batch 在引擎流上一定排在旧 batch 之后。
        if self._defer_slot_free:
            self._deferred_slots.append(req.table_idx)
        else:
            self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _release_deferred_slots(self) -> None:
        for slot in self._deferred_slots:
            self.table_manager.free(slot)
        self._deferred_slots = []

    # ------------------------------------------------------------------ 调度与前向
    def _schedule_next_batch(self) -> ForwardInput | None:
        # prefill 优先：有等待的请求就先做 prefill，否则做 decode
        batch = (self.prefill_manager.schedule_next_batch(self.prefill_budget)
                 or self.decode_manager.schedule_next_batch())
        return self._prepare_batch(batch) if batch else None

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(batch, self.engine.sampler.prepare(batch), input_mapping, write_mapping)

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, write_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]  # 在设备上取输入，不经过 CPU
        output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[write_mapping] = output.next_tokens_gpu  # 采样结果直接写回，作为下一轮输入
        self.decode_manager.filter_reqs(batch.reqs)
        return output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    """每个请求本轮要算的位置：[cached_len, device_len)。"""
    host = torch.empty(sum(r.extend_len for r in batch.padded_reqs), dtype=torch.int32,
                       pin_memory=pin(device))
    offset = 0
    for req in batch.padded_reqs:
        n = req.extend_len
        torch.arange(req.cached_len, req.device_len, dtype=torch.int32,
                     out=host[offset : offset + n])
        offset += n
    return host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    rows = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=pin(device))
    offset = 0
    for req in batch.padded_reqs:
        rows[offset : offset + req.extend_len].fill_(req.table_idx)
        offset += req.extend_len
    return rows.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    """采样出的 token 写到 token_pool[table_idx, device_len]，也就是下一轮的输入位置。

    列号为 -1 表示不写：分块中的请求和已经不能再生成的请求（写 -1 相当于写到最后一列，无害）。
    注意 device_len 在 forward_batch 里会加 1，这里读取的是加 1 之前的值，正好是新 token 的位置。
    """
    rows = torch.tensor([r.table_idx for r in batch.reqs], dtype=torch.int64, pin_memory=pin(device))
    cols = torch.tensor([r.device_len if r.can_decode else -1 for r in batch.reqs],
                        dtype=torch.int64, pin_memory=pin(device))
    return rows.to(device, non_blocking=True), cols.to(device, non_blocking=True)
