"""minitl：Triton 编程模型的 numpy 模拟器（没有 NVIDIA GPU 时代替 triton / triton.language）。

题目里照常写

    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        ...

判题器在没有真 Triton 的环境里把 triton、triton.language 换成这个模块：每个 program 依次执行，
块内的运算用 numpy 向量化完成。和真 Triton 一样，tl.arange 的长度必须是 2 的幂、越界的 load/store
必须用 mask 屏蔽、被屏蔽的元素如果没给 other 就是未定义值（这里用 NaN 表示）。
另外会统计 load / store 的次数和字节数，用来检验"融合"（例如 softmax 只读一遍输入）。
"""

from __future__ import annotations

import builtins as _b
import itertools

import numpy as np


class TritonError(Exception):
    pass


class constexpr:  # 只用作类型标注：BLOCK: tl.constexpr
    pass


float16, float32, float64 = np.float16, np.float32, np.float64
int8, int16, int32, int64, uint8, uint32 = np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32
bfloat16 = np.float32   # numpy 没有 bfloat16，用 float32 代替


class Tensor(np.ndarray):
    """块（block）的值：就是 numpy 数组，多了 Triton 的 .to(dtype)。"""

    def to(self, dtype):
        return np.asarray(self).astype(dtype).view(Tensor)


def _t(x):
    return np.asarray(x).view(Tensor)


class Pointer:
    """指针（或指针块）：底层缓冲区 + 元素偏移。"""

    def __init__(self, buf: np.ndarray, offset=0, name="ptr"):
        self.buf, self.offset, self.name = buf, offset, name

    def __add__(self, other):
        return Pointer(self.buf, self.offset + np.asarray(other), self.name)

    __radd__ = __add__

    def __sub__(self, other):
        return Pointer(self.buf, self.offset - np.asarray(other), self.name)

    @property
    def dtype(self):
        return self.buf.dtype


class Stats:
    def __init__(self):
        self.programs = self.loads = self.stores = self.load_bytes = self.store_bytes = self.dots = 0

    def __repr__(self):
        return (f"Stats(programs={self.programs}, loads={self.loads}, stores={self.stores}, "
                f"load_bytes={self.load_bytes}, store_bytes={self.store_bytes}, dots={self.dots})")


class _Ctx:
    pid = (0, 0, 0)
    grid = (1, 1, 1)
    stats = Stats()


_ctx = _Ctx()
last_stats = None


def program_id(axis: int):
    return _ctx.pid[axis]


def num_programs(axis: int):
    return _ctx.grid[axis]


def arange(start: int, end: int):
    n = int(end) - int(start)
    if n <= 0 or n & (n - 1):
        raise TritonError(f"tl.arange({start}, {end}) 的长度 {n} 必须是 2 的幂")
    return _t(np.arange(start, end, dtype=np.int32))


def _mask(ptr: Pointer, mask):
    off = np.asarray(ptr.offset)
    m = np.ones(off.shape, dtype=bool) if mask is None else np.broadcast_to(np.asarray(mask, dtype=bool), off.shape)
    return off, m


def _check_bounds(ptr: Pointer, off, m, what):
    sel = off[m] if off.ndim else (off if m else np.array([], dtype=np.int64))
    sel = np.atleast_1d(sel)
    if sel.size and (sel.min() < 0 or sel.max() >= ptr.buf.size):
        bad = int(sel.max() if sel.max() >= ptr.buf.size else sel.min())
        raise TritonError(f"{what} 越界：访问 {ptr.name} 的第 {bad} 个元素，而它只有 {ptr.buf.size} 个"
                          f"（是不是忘了用 mask 屏蔽越界的位置？）")


def load(ptr: Pointer, mask=None, other=None, **_):
    if not isinstance(ptr, Pointer):
        raise TritonError("tl.load 的第一个参数必须是指针（例如 x_ptr + offs）")
    off, m = _mask(ptr, mask)
    _check_bounds(ptr, off, m, "tl.load")
    fill = other if other is not None else (np.nan if ptr.buf.dtype.kind == "f" else 0)
    out = np.full(off.shape, fill, dtype=ptr.buf.dtype)
    if off.ndim:
        out[m] = ptr.buf[off[m]]
    elif m:
        out = np.asarray(ptr.buf[int(off)])
    _ctx.stats.loads += 1
    _ctx.stats.load_bytes += int(m.sum()) * ptr.buf.itemsize
    return _t(out)


def store(ptr: Pointer, value, mask=None, **_):
    if not isinstance(ptr, Pointer):
        raise TritonError("tl.store 的第一个参数必须是指针（例如 out_ptr + offs）")
    off, m = _mask(ptr, mask)
    _check_bounds(ptr, off, m, "tl.store")
    val = np.broadcast_to(np.asarray(value), off.shape).astype(ptr.buf.dtype)
    if off.ndim:
        ptr.buf[off[m]] = val[m]
    elif m:
        ptr.buf[int(off)] = val
    _ctx.stats.stores += 1
    _ctx.stats.store_bytes += int(m.sum()) * ptr.buf.itemsize


def zeros(shape, dtype=np.float32):
    return _t(np.zeros(shape, dtype=dtype))


def full(shape, value, dtype=np.float32):
    return _t(np.full(shape, value, dtype=dtype))


def _reduce(fn, x, axis=None, keep_dims=False):
    return _t(fn(np.asarray(x), axis=axis, keepdims=keep_dims))


def sum(x, axis=None, keep_dims=False):  # noqa: A001  与 tl.sum 同名
    return _reduce(np.sum, x, axis, keep_dims)


def max(x, axis=None, keep_dims=False):  # noqa: A001
    return _reduce(np.max, x, axis, keep_dims)


def min(x, axis=None, keep_dims=False):  # noqa: A001
    return _reduce(np.min, x, axis, keep_dims)


def argmax(x, axis):
    return _t(np.argmax(np.asarray(x), axis=axis).astype(np.int32))


def exp(x):
    return _t(np.exp(x))


def exp2(x):
    return _t(np.exp2(x))


def log(x):
    return _t(np.log(x))


def sqrt(x):
    return _t(np.sqrt(x))


def rsqrt(x):
    return _t(1.0 / np.sqrt(x))


def abs(x):  # noqa: A001
    return _t(np.abs(x))


def sigmoid(x):
    return _t(1.0 / (1.0 + np.exp(-np.asarray(x))))


def maximum(a, b):
    return _t(np.maximum(a, b))


def minimum(a, b):
    return _t(np.minimum(a, b))


def where(cond, a, b):
    return _t(np.where(cond, a, b))


def trans(x):
    return _t(np.asarray(x).T)


def reshape(x, shape):
    return _t(np.reshape(x, shape))


def cdiv(a, b):
    return (a + b - 1) // b


def dot(a, b, acc=None, **_):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise TritonError("tl.dot 的两个参数都必须是二维块")
    if _b.min(*a.shape, b.shape[1]) < 16:
        raise TritonError(f"tl.dot 要求每一维至少 16，这里是 {a.shape} @ {b.shape}")
    _ctx.stats.dots += 1
    out = a.astype(np.float32) @ b.astype(np.float32)
    return _t(out if acc is None else np.asarray(acc) + out)


def static_range(*args):
    return range(*args)


def multiple_of(x, _):
    return x


def max_contiguous(x, _):
    return x


def device_assert(cond, msg=""):
    if not np.all(cond):
        raise TritonError(f"device_assert 失败：{msg}")


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__

    def __getitem__(self, grid):
        def launcher(*args, **kwargs):
            meta = dict(kwargs)
            g = grid(meta) if callable(grid) else grid
            g = tuple(int(v) for v in (g if isinstance(g, (tuple, list)) else (g,)))
            g = g + (1,) * (3 - len(g))
            conv = []
            names = self.fn.__code__.co_varnames[: self.fn.__code__.co_argcount]
            for name, a in zip(names, args):
                if isinstance(a, np.ndarray):
                    if not a.flags["C_CONTIGUOUS"]:
                        raise TritonError(f"参数 {name} 必须是连续内存（C_CONTIGUOUS）")
                    conv.append(Pointer(a.reshape(-1), 0, name))
                else:
                    conv.append(a)
            stats = Stats()
            _ctx.stats, _ctx.grid = stats, g
            for pid in itertools.product(range(g[0]), range(g[1]), range(g[2])):
                _ctx.pid = pid
                stats.programs += 1
                self.fn(*conv, **kwargs)
            global last_stats
            last_stats = stats
            return stats

        return launcher

    def __call__(self, *args, **kwargs):
        raise TritonError(f"{self.__name__} 要用 {self.__name__}[grid](...) 启动")


def jit(fn=None, **_):
    if fn is None:
        return lambda f: JITFunction(f)
    return JITFunction(fn)
