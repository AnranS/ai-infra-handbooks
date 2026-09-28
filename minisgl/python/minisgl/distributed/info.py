from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    size: int

    def __post_init__(self) -> None:
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


# 每个 TP 进程只有一份 TP 信息，设成进程级的全局变量，各层构造时直接读取
_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


def reset_tp_info() -> None:
    """测试用：同一进程里重建引擎前清空。"""
    global _TP_INFO
    _TP_INFO = None
