"""测试里常用的检查函数：失败时给出清楚的中文说明（期望值、实际值、差在哪里）。

在测试文件里 `from checker import check, check_close, raises, time_limit`。
"""

from __future__ import annotations

import contextlib
import math
import sys
import time

IN_BROWSER = sys.platform == "emscripten"
# 浏览器里的 Python（WebAssembly）比本地慢，时间限制相应放宽
SPEED_FACTOR = 3.0 if IN_BROWSER else 1.0


class Skip(Exception):
    """当前环境跑不了这个用例（例如没有 GPU），判题时记为"跳过"。"""


def skip(reason: str):
    raise Skip(reason)


def _is_array(x) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and not isinstance(x, (int, float, complex))


def _s(x, limit: int = 400) -> str:
    text = repr(x)
    if _is_array(x) and getattr(x, "size", 0) > 16:
        text = f"<{type(x).__name__} shape={tuple(x.shape)} dtype={x.dtype}>\n{x!r}"
    return text if len(text) <= limit else text[:limit] + " …"


def _to_numpy(x):
    if hasattr(x, "detach"):          # torch.Tensor
        x = x.detach().float().cpu().numpy() if str(getattr(x, "dtype", "")).startswith("torch.bfloat16") \
            else x.detach().cpu().numpy()
    return x


def check(actual, expected, what: str = "结果") -> None:
    """精确相等（数组逐元素相等、形状也要相同）。"""
    actual, expected = _to_numpy(actual), _to_numpy(expected)
    if _is_array(actual) or _is_array(expected):
        import numpy as np

        a, e = np.asarray(actual), np.asarray(expected)
        if a.shape != e.shape:
            raise AssertionError(f"{what}：形状不对，期望 {e.shape}，实际 {a.shape}")
        if not np.array_equal(a, e):
            bad = np.argwhere(a != e)
            i = tuple(int(v) for v in bad[0])
            raise AssertionError(f"{what}：有 {len(bad)} 个元素不对，第一个在下标 {i}：期望 {e[i]!r}，实际 {a[i]!r}\n"
                                 f"  期望：{_s(e)}\n  实际：{_s(a)}")
        return
    if actual != expected:
        raise AssertionError(f"{what}\n  期望：{_s(expected)}\n  实际：{_s(actual)}")


def check_close(actual, expected, rtol: float = 1e-5, atol: float = 1e-6, what: str = "结果") -> None:
    """近似相等：|实际 - 期望| <= atol + rtol * |期望|。"""
    actual, expected = _to_numpy(actual), _to_numpy(expected)
    if _is_array(actual) or _is_array(expected) or isinstance(actual, (list, tuple)):
        import numpy as np

        a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        if a.shape != e.shape:
            raise AssertionError(f"{what}：形状不对，期望 {e.shape}，实际 {a.shape}")
        if np.isnan(a).any() and not np.isnan(e).any():
            i = tuple(int(v) for v in np.argwhere(np.isnan(a))[0])
            raise AssertionError(f"{what}：出现了 NaN，第一个在下标 {i}")
        diff = np.abs(a - e)
        bad = diff > atol + rtol * np.abs(e)
        if bad.any():
            i = tuple(int(v) for v in np.argwhere(bad)[0])
            raise AssertionError(f"{what}：误差太大，{int(bad.sum())} 个元素超出容差，最大误差 {diff.max():.3g}；"
                                 f"下标 {i} 处期望 {e[i]:.6g}，实际 {a[i]:.6g}（容差 rtol={rtol}, atol={atol}）")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f"{what}：期望 {expected!r}，实际 {actual!r}（容差 rtol={rtol}, atol={atol}）")


@contextlib.contextmanager
def raises(exc_type, what: str = "这个调用"):
    """断言代码块抛出指定类型的异常。"""
    try:
        yield
    except exc_type:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"{what}：应该抛出 {exc_type.__name__}，实际抛出了 {type(e).__name__}: {e}") from None
    raise AssertionError(f"{what}：应该抛出 {exc_type.__name__}，但没有抛出任何异常")


@contextlib.contextmanager
def time_limit(seconds: float, what: str = "这段代码"):
    """断言代码块在限定时间内完成（浏览器里自动放宽）。"""
    limit = seconds * SPEED_FACTOR
    start = time.perf_counter()
    yield
    used = time.perf_counter() - start
    if used > limit:
        raise AssertionError(f"{what}：用了 {used:.2f} 秒，超过限制 {limit:.2f} 秒：算法的复杂度可能太高")


# ---------- 运行环境 ----------

def torch_device() -> str:
    """本地 PyTorch 题用的设备：WSL2 / Linux 上的 NVIDIA GPU 用 cuda，Apple Silicon 用 mps，否则 cpu。"""
    try:
        import torch
    except ImportError:
        skip("这道题需要 PyTorch：请在本地环境运行（见 practice/README.md）")
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def need_torch():
    if IN_BROWSER:
        skip("浏览器里没有 PyTorch：请在本地运行 python practice/judge.py <题号>")
    return torch_device()


def need_cuda():
    dev = need_torch()
    if dev != "cuda":
        skip("这个用例需要 NVIDIA GPU（例如 WSL2 + CUDA），当前设备是 " + dev)
