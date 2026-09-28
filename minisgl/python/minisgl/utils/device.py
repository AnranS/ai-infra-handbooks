"""CPU / CUDA 的最小抽象层（官方实现没有这一层，它只支持 CUDA）。

官方代码直接用 torch.cuda.Stream、torch.cuda.Event 和 pin_memory。为了在没有 GPU 的机器上
也能跑通整个引擎，我们把这几样东西包一层：在 CUDA 上就是原来的对象；在 CPU 上，所有操作
本来就是按顺序同步执行的，stream 和 event 都退化成什么也不做的空对象。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ContextManager

import torch


def is_cuda(device: torch.device) -> bool:
    return torch.device(device).type == "cuda"


class NullStream:
    """CPU 上的"流"：没有异步执行，等待别的流是空操作。"""

    def wait_stream(self, other: Any) -> None:
        pass

    def synchronize(self) -> None:
        pass


class NullEvent:
    """CPU 上的"事件"：记录和等待都是空操作（record 时计算早已完成）。"""

    def record(self, stream: Any = None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True


def create_stream(device: torch.device) -> Any:
    return torch.cuda.Stream(device=device) if is_cuda(device) else NullStream()


def create_event(device: torch.device) -> Any:
    return torch.cuda.Event() if is_cuda(device) else NullEvent()


def set_stream(stream: Any) -> None:
    if isinstance(stream, torch.cuda.Stream):
        torch.cuda.set_stream(stream)


def stream_ctx(stream: Any) -> ContextManager:
    return torch.cuda.stream(stream) if isinstance(stream, torch.cuda.Stream) else nullcontext()


def pin(device: torch.device) -> bool:
    """只有在 CUDA 上才需要（也才能）使用锁页内存做异步拷贝。"""
    return is_cuda(device)


def synchronize(device: torch.device) -> None:
    if is_cuda(device):
        torch.cuda.synchronize(device)


def get_free_memory(device: torch.device) -> int:
    if is_cuda(device):
        return torch.cuda.mem_get_info(device)[0]
    import psutil

    return psutil.virtual_memory().available
