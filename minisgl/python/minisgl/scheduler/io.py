from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import torch
import torch.distributed as dist
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """调度器的收发：从 tokenizer 收请求、把生成的 token 发给 detokenizer。

    张量并行时只有 rank 0 连 tokenizer。rank 0 把收到的原始字节原样 PUB 给其他 rank，
    并先用一次 broadcast 告诉大家"这一轮有几条消息"，保证所有 rank 处理完全相同的消息序列、
    做出完全相同的调度决策——各 rank 的 KV 池和 page table 因此始终一致。
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: Any):
        tp_info = config.tp_info
        self.tp_cpu_group = tp_cpu_group
        if config.offline_mode:  # 离线模式：由 LLM 子类实现收发
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return
        if tp_info.is_primary():
            self._recv_from_tokenizer = ZmqPullQueue(
                config.zmq_backend_addr, create=True, decoder=BaseBackendMsg.decoder)
            self._send_into_tokenizer = ZmqPushQueue(
                config.zmq_detokenizer_addr, create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder)
        recv, send = self._recv_msg_single_rank, self._reply_tokenizer_rank0
        if tp_info.size > 1:
            if tp_info.is_primary():
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder)
                # 等其他 rank 的订阅都生效：否则第一条广播可能被丢掉，那个 rank 会永远等下去
                self._send_into_ranks.wait_for_subscribers(tp_info.size - 1)
            else:
                recv, send = self._recv_msg_multi_rank1, self._reply_tokenizer_rank1
                self._recv_from_rank0 = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr, create=False, decoder=BaseBackendMsg.decoder)
        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self) -> None:
        raise NotImplementedError

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        raise NotImplementedError

    def sync_all_ranks(self) -> None:
        if self.tp_cpu_group is not None:
            dist.barrier(group=self.tp_cpu_group)

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        msgs: List[BaseBackendMsg] = []
        if blocking:  # 没有活可干：阻塞等待第一条消息
            self.run_when_idle()
            msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():  # 再把已经到达的消息一次取完
            msgs.append(self._recv_from_tokenizer.get())
        return msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            msgs.append(self._recv_from_tokenizer.decode(raw))
        raws: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            raws.append(self._recv_from_tokenizer.get_raw())
        count = torch.tensor(len(raws))
        dist.broadcast(count, src=0, group=self.tp_cpu_group)  # 告诉其他 rank 这一轮收几条
        for raw in raws:
            self._send_into_ranks.put_raw(raw)
            msgs.append(self._recv_from_tokenizer.decode(raw))
        return msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            msgs.append(self._recv_from_rank0.get())
        count = torch.tensor(-1)
        dist.broadcast(count, src=0, group=self.tp_cpu_group)
        for _ in range(int(count)):
            msgs.append(self._recv_from_rank0.get())
        return msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        if len(reply) == 1:
            self._send_into_tokenizer.put(reply[0])
        elif len(reply) > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore[arg-type]

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        pass  # 只有 rank 0 回复
