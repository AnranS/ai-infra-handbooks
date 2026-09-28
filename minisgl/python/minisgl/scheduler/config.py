from __future__ import annotations

import os
from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192  # 一个 prefill batch 最多处理多少 token（分块 prefill 的块大小）
    cache_type: str = "radix"
    offline_mode: bool = False  # True：不起 ZMQ，由 LLM 类直接喂请求、收结果

    # IPC 地址带上启动进程的 pid，同一台机器上起多个服务也不会冲突
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:  # tokenizer -> scheduler
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:  # scheduler -> detokenizer
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:  # rank 0 -> 其他 rank
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
