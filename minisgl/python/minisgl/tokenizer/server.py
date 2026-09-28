from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer

from .detokenize import DetokenizeManager
from .tokenize import TokenizeManager


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    return msg.data if isinstance(msg, BatchTokenizerMsg) else [msg]


def _pack(items: list, batch_cls):
    """多条消息打包成一个 Batch 消息发送；只有一条时直接发送它本身。"""
    return items[0] if len(items) == 1 else batch_cls(data=items)


@torch.inference_mode()
def tokenize_worker(*, tokenizer_path: str, addr: str, create: bool, backend_addr: str,
                    frontend_addr: str, local_bs: int, tokenizer_id: int = -1,
                    ack_queue: mp.Queue | None = None) -> None:
    """tokenizer 进程的主循环。同一个进程既能分词（前端 → 调度器），也能反分词（调度器 → 前端）。"""
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")
    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)
    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending = _unwrap_msg(recv_listener.get())
            while len(pending) < local_bs and not recv_listener.empty():
                pending.extend(_unwrap_msg(recv_listener.get()))
            logger.debug(f"Received {len(pending)} messages")
            detok = [m for m in pending if isinstance(m, DetokenizeMsg)]
            tok = [m for m in pending if isinstance(m, TokenizeMsg)]
            abort = [m for m in pending if isinstance(m, AbortMsg)]
            if detok:
                texts = detokenize_manager.detokenize(detok)
                replies = [UserReply(uid=m.uid, incremental_output=t, finished=m.finished)
                           for m, t in zip(detok, texts, strict=True)]
                send_frontend.put(_pack(replies, BatchFrontendMsg))
            if tok:
                tensors = tokenize_manager.tokenize(tok)
                reqs = [UserMsg(uid=m.uid, input_ids=t, sampling_params=m.sampling_params)
                        for m, t in zip(tok, tensors, strict=True)]
                send_backend.put(_pack(reqs, BatchBackendMsg))
            if abort:
                send_backend.put(_pack([AbortBackendMsg(uid=m.uid) for m in abort],
                                       BatchBackendMsg))
    except KeyboardInterrupt:
        pass
