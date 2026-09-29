"""Triton 题的运行环境切换：有 NVIDIA GPU 和真 Triton 时用真的，否则用 minitl 模拟器。

测试里用 to_dev / empty_like / to_host 在"numpy 数组（模拟器）"和"CUDA 上的 torch.Tensor（真 Triton）"之间转换，
所以同一份题解在浏览器、Mac 和 WSL2 上都能跑。
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

REAL = False


def install(prefer_real: bool = True) -> str:
    """在导入题解之前调用：决定 `import triton` 拿到的是真 Triton 还是模拟器。"""
    global REAL
    if prefer_real and os.environ.get("PRACTICE_TRITON", "auto") != "emulate" and sys.platform != "emscripten":
        try:
            import torch
            import triton  # noqa: F401

            if torch.cuda.is_available():
                REAL = True
                return "triton"
        except ImportError:
            pass
    import minitl

    shim = types.ModuleType("triton")
    shim.jit = minitl.jit
    shim.cdiv = minitl.cdiv
    shim.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()
    shim.language = minitl
    shim.__emulated__ = True
    sys.modules["triton"] = shim
    sys.modules["triton.language"] = minitl
    REAL = False
    return "minitl"


def to_dev(x: np.ndarray):
    if REAL:
        import torch

        return torch.from_numpy(np.ascontiguousarray(x)).cuda()
    return np.ascontiguousarray(x).copy()


def empty_like(x):
    if REAL:
        import torch

        return torch.empty_like(x)
    return np.full_like(x, np.nan) if x.dtype.kind == "f" else np.zeros_like(x)


def empty(shape, like, dtype=None):
    """分配一块和 like 在同一设备上的"脏"内存（模拟器下是 numpy 数组，真 Triton 下是 CUDA 张量）。"""
    if REAL:
        import torch

        return torch.empty(shape, dtype=dtype or like.dtype, device=like.device)
    dt = np.dtype(dtype or like.dtype)
    return np.full(shape, np.nan, dtype=dt) if dt.kind == "f" else np.zeros(shape, dtype=dt)


def to_host(x) -> np.ndarray:
    if REAL:
        return x.detach().cpu().numpy()
    return np.asarray(x)


def last_stats():
    """模拟器下返回最近一次 launch 的 load/store 统计；真 Triton 下返回 None。"""
    if REAL:
        return None
    import minitl

    return minitl.last_stats
