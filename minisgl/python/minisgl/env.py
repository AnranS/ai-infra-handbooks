"""以 MINISGL_ 开头的环境变量。和官方一样用一个单例集中声明，读取时自动转换类型。"""

from __future__ import annotations

import os
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class EnvVar(Generic[T]):
    def __init__(self, default: T, fn: Callable[[str], T]):
        self.value = default
        self.fn = fn

    def load(self, name: str) -> None:
        raw = os.getenv(name)
        if raw is not None:
            self.value = self.fn(raw)

    def __bool__(self) -> bool:
        return bool(self.value)


def _to_bool(x: str) -> bool:
    return x.lower() in ("1", "true", "yes")


class _Env:
    SHELL_MAX_TOKENS = EnvVar(2048, int)
    SHELL_TEMPERATURE = EnvVar(0.6, float)
    DISABLE_OVERLAP_SCHEDULING = EnvVar(False, _to_bool)

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        for name in dir(self):
            var = getattr(self, name)
            if isinstance(var, EnvVar):
                var.load("MINISGL_" + name)


ENV = _Env()
