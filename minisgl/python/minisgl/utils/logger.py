from __future__ import annotations

import logging
import os
import sys


class _Logger(logging.Logger):
    """在标准 Logger 上加 *_rank0 方法：张量并行时只让 rank 0 打印，避免每条日志出现 N 遍。"""

    def _rank0(self, level: int, msg, *args, **kwargs) -> None:
        from minisgl.distributed import try_get_tp_info

        info = try_get_tp_info()
        if info is None or info.is_primary():
            self.log(level, msg, *args, **kwargs)

    def info_rank0(self, msg, *args, **kwargs) -> None:
        self._rank0(logging.INFO, msg, *args, **kwargs)

    def warning_rank0(self, msg, *args, **kwargs) -> None:
        self._rank0(logging.WARNING, msg, *args, **kwargs)

    def debug_rank0(self, msg, *args, **kwargs) -> None:
        self._rank0(logging.DEBUG, msg, *args, **kwargs)


def init_logger(name: str, suffix: str = "") -> _Logger:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    old_cls = logging.getLoggerClass()
    logging.setLoggerClass(_Logger)
    try:
        logger = logging.getLogger(name)
    finally:
        logging.setLoggerClass(old_cls)
    assert isinstance(logger, _Logger)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        tag = f"|{suffix}" if suffix else ""
        handler.setFormatter(
            logging.Formatter(f"[%(asctime)s{tag}] %(levelname)-7s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger
