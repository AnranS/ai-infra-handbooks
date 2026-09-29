"""gpusim：用 Python 写的 SIMT 模拟器，在没有 NVIDIA GPU 的地方写、测 CUDA 风格的 kernel。

写法和 CUDA C++ 一一对应：

    import gpusim as gs

    @gs.kernel
    def add(t, a, b, c, n):                   # t 相当于 CUDA 里的内置变量
        i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
        if i < n:
            c[i] = a[i] + b[i]

    a = gs.to_device(np.arange(1000, dtype=np.float32))
    ...
    stats = add[gs.cdiv(n, 256), 256](a, b, c, n)   # add<<<grid, block>>>(...)
    c.copy_to_host()

对应关系：
    __shared__ float s[32][33]    ->  s = t.shared("s", (32, 33))
    __syncthreads()               ->  yield t.syncthreads()
    __shfl_down_sync(FULL, v, d)  ->  v2 = yield t.shfl_down(v, d)     （还有 shfl_up / shfl_xor / shfl / ballot / syncwarp）
    atomicAdd(&a[i], v)           ->  old = t.atomic_add(a, i, v)

带 yield 的 kernel 是一个生成器：每个线程是一个协程，所有线程都走到同一个 yield（屏障或 warp 操作）
之后才一起继续，所以屏障、warp shuffle 的语义与 GPU 一致。模拟器还会：

- 检查越界访问、读未初始化的内存、__syncthreads 不在所有线程上执行（例如提前 return）；
- 检测数据竞争：同一阶段（两个屏障之间）不同线程对同一地址一读一写或都写（原子操作除外）；
- 统计访存：每个 warp 的每条访存指令触及多少个 32 字节扇区（合并访问的程度）、共享内存的 bank conflict。

返回的 Stats 可以用来检验优化效果，例如"转置的全局内存扇区数不超过 N"、"共享内存没有 bank conflict"。
"""

from __future__ import annotations

import inspect
import math

import numpy as np

WARP_SIZE = 32
SECTOR = 32               # 全局内存事务的粒度（字节）
BANKS = 32
MAX_THREADS_PER_BLOCK = 1024
MAX_SHARED_BYTES = 48 * 1024


class KernelError(Exception):
    """kernel 的写法违反了 CUDA 的规则（越界、竞争、屏障不一致等）。"""


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class Dim3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=1, y=1, z=1):
        self.x, self.y, self.z = int(x), int(y), int(z)

    def __repr__(self):
        return f"dim3({self.x}, {self.y}, {self.z})"

    @property
    def size(self):
        return self.x * self.y * self.z


def _dim3(v) -> Dim3:
    if isinstance(v, Dim3):
        return v
    if isinstance(v, (tuple, list)):
        return Dim3(*v)
    return Dim3(int(v))


class Stats:
    """一次 launch 的访存统计。"""

    FIELDS = ("global_load_requests", "global_store_requests", "global_load_sectors", "global_store_sectors",
              "global_load_bytes", "global_store_bytes", "shared_load_requests", "shared_store_requests",
              "shared_wavefronts", "bank_conflicts", "syncthreads", "warp_ops", "atomics", "global_atomics",
              "shared_atomics", "threads", "blocks")

    def __init__(self):
        for f in self.FIELDS:
            setattr(self, f, 0)
        self.arrays: dict = {}    # 按数组名分别统计全局内存访问

    def array(self, name):
        """某个全局数组的访存统计：load_sectors、load_bytes、store_sectors、store_bytes、load_efficiency。"""
        a = dict(self.arrays.get(name, {"load_sectors": 0, "load_bytes": 0, "store_sectors": 0, "store_bytes": 0}))
        a["load_efficiency"] = a["load_bytes"] / (a["load_sectors"] * SECTOR) if a["load_sectors"] else 1.0
        a["store_efficiency"] = a["store_bytes"] / (a["store_sectors"] * SECTOR) if a["store_sectors"] else 1.0
        return a

    @property
    def global_sectors(self):
        return self.global_load_sectors + self.global_store_sectors

    @property
    def global_requests(self):
        return self.global_load_requests + self.global_store_requests

    @property
    def load_efficiency(self):
        """有用的字节 / 实际搬运的字节（1.0 表示完全合并）。"""
        return self.global_load_bytes / (self.global_load_sectors * SECTOR) if self.global_load_sectors else 1.0

    @property
    def store_efficiency(self):
        return self.global_store_bytes / (self.global_store_sectors * SECTOR) if self.global_store_sectors else 1.0

    def as_dict(self):
        d = {f: getattr(self, f) for f in self.FIELDS}
        d.update(load_efficiency=round(self.load_efficiency, 3), store_efficiency=round(self.store_efficiency, 3))
        return d

    def __repr__(self):
        return ("Stats(全局读 {0} 次请求 / {1} 个扇区，全局写 {2} 次请求 / {3} 个扇区，读效率 {4:.0%}，写效率 {5:.0%}，"
                "共享内存 {6} 次读 / {7} 次写，bank conflict {8}，__syncthreads {9} 次)").format(
            self.global_load_requests, self.global_load_sectors, self.global_store_requests,
            self.global_store_sectors, self.load_efficiency, self.store_efficiency, self.shared_load_requests,
            self.shared_store_requests, self.bank_conflicts, self.syncthreads)


# ---------------------------------------------------------------- 内存

_next_addr = [1 << 20]
_CUR = None               # 正在执行的线程
_TOUCHED: set = set()     # 本次 launch 访问过的数组（launch 结束时清空竞争检测的记录）


def _alloc_addr(nbytes: int) -> int:
    addr = _next_addr[0]
    _next_addr[0] += (nbytes + 511) // 256 * 256 + 256
    return addr


class _Rec:
    __slots__ = ("gtid", "block", "bepoch", "warp", "wepoch", "atomic", "who")

    def __init__(self, th, atomic):
        self.gtid, self.block, self.bepoch = th.gtid, th.block_linear, th.blk.bepoch
        self.warp, self.wepoch, self.atomic = th.warp_gid, th.blk.wepoch[th.warp_in_block], atomic
        self.who = th


def _ordered(a: _Rec, b: _Rec) -> bool:
    if a.block != b.block:
        return False
    return a.bepoch != b.bepoch or (a.warp == b.warp and a.wepoch != b.wepoch)


class DeviceArray:
    """设备上的数组。kernel 里只能按元素访问：a[i]、a[i, j]。"""

    space = "global"

    def __init__(self, data: np.ndarray, name: str = "array", initialized: bool = True):
        self.data = np.ascontiguousarray(data)
        self.flat = self.data.reshape(-1)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.itemsize = self.data.itemsize
        self.name = name
        self.base = _alloc_addr(self.data.nbytes)
        self.init = None if initialized else np.zeros(self.flat.shape, dtype=bool)
        self._strides = [int(np.prod(self.shape[i + 1:])) for i in range(len(self.shape))]
        self._reset_tracking()

    # 主机端接口
    def copy_to_host(self) -> np.ndarray:
        return self.data.copy()

    def __len__(self):
        return self.shape[0]

    @property
    def size(self):
        return self.flat.size

    def __repr__(self):
        return f"<{type(self).__name__} {self.name} shape={self.shape} dtype={self.dtype}>"

    def _reset_tracking(self):
        self._last_write: dict = {}
        self._reads: dict = {}

    def _int(self, k) -> int:
        if type(k) is int:
            return k
        if isinstance(k, (np.integer, bool)):
            return int(k)
        if isinstance(k, slice):
            raise KernelError(f"kernel 里只能按元素访问 {self.name}，不能切片")
        if isinstance(k, (float, np.floating)):
            raise KernelError(f"下标必须是整数，{self.name} 却用 {k!r} 访问（是不是用了 / 而不是 //？）")
        raise KernelError(f"{self.name} 的下标类型不对：{k!r}")

    def _index(self, key) -> int:
        if isinstance(key, tuple):
            if len(key) != len(self.shape):
                raise KernelError(f"{self.name} 是 {len(self.shape)} 维数组，却用 {len(key)} 个下标访问")
            idx = 0
            for k, n, st in zip(key, self.shape, self._strides):
                k = self._int(k)
                if k < 0 or k >= n:
                    raise KernelError(f"越界访问：{self.name}{list(key)}，形状是 {self.shape}")
                idx += k * st
            return idx
        k = self._int(key)
        if k < 0 or k >= self.flat.size:
            if len(self.shape) == 1:
                raise KernelError(f"越界访问：{self.name}[{k}]，数组长度是 {self.shape[0]}")
            raise KernelError(f"越界访问：{self.name}[{k}]（按一维展开访问，共 {self.flat.size} 个元素）")
        return k

    def _where(self, idx):
        if len(self.shape) == 1:
            return f"{self.name}[{idx}]"
        return f"{self.name}{list(np.unravel_index(idx, self.shape))}"

    def _record(self, idx: int, write: bool, atomic: bool = False):
        th = _CUR
        if th is None:
            return
        if th.pending_sync:
            raise KernelError("t.syncthreads() 要写成 yield t.syncthreads()，否则屏障不会生效")
        _TOUCHED.add(self)
        th.log.append((self, write, atomic, self.base + idx * self.itemsize))
        rec = _Rec(th, atomic)
        lw = self._last_write.get(idx)
        if lw is not None and lw.gtid != rec.gtid and not (lw.atomic and atomic) and not _ordered(lw, rec):
            self._race(idx, lw, rec, "写" if write else "读")
        if write:
            reads = self._reads.pop(idx, None)
            if reads:
                for r in reads.values():
                    if r.gtid != rec.gtid and not (r.atomic and atomic) and not _ordered(r, rec):
                        self._race(idx, r, rec, "写", other="读")
            self._last_write[idx] = rec
        else:
            self._reads.setdefault(idx, {})[rec.gtid] = rec
            if self.init is not None and not self.init[idx]:
                hint = "；如果它应该由别的线程先写好，说明少了 yield t.syncthreads()" if self.space == "shared" else ""
                raise KernelError(f"读了还没有写入过的内存 {self._where(idx)}（{th.describe()}）{hint}")
        if write and self.init is not None:
            self.init[idx] = True

    def _race(self, idx, a: _Rec, b: _Rec, kind, other="写"):
        hint = "是不是少了 yield t.syncthreads()？" if a.block == b.block else \
            "不同 block 之间没有同步手段，需要原子操作或拆成两个 kernel。"
        raise KernelError(f"数据竞争：{b.who.describe()} {kind} {self._where(idx)} 时，"
                          f"{a.who.describe()} 在同一阶段{other}过它。{hint}")

    def __getitem__(self, key):
        idx = self._index(key)
        self._record(idx, False)
        return self.flat[idx]

    def __setitem__(self, key, value):
        idx = self._index(key)
        self._record(idx, True)
        self.flat[idx] = value


class SharedArray(DeviceArray):
    space = "shared"


def to_device(x, name: str = "array") -> DeviceArray:
    return DeviceArray(np.array(x, copy=True), name)


def zeros(shape, dtype=np.float32, name: str = "array") -> DeviceArray:
    return DeviceArray(np.zeros(shape, dtype=dtype), name)


def empty(shape, dtype=np.float32, name: str = "array") -> DeviceArray:
    """像 cudaMalloc 一样：内容未初始化，kernel 读之前必须先写。"""
    data = np.full(shape, np.nan, dtype=dtype) if np.dtype(dtype).kind == "f" else np.full(shape, -7, dtype=dtype)
    return DeviceArray(data, name, initialized=False)


# ---------------------------------------------------------------- 线程与 warp 操作

class _Sync:
    kind = "sync"


class _WarpOp:
    __slots__ = ("kind", "value", "arg", "width")

    def __init__(self, kind, value=None, arg=0, width=WARP_SIZE):
        self.kind, self.value, self.arg, self.width = kind, value, arg, width


class _Block:
    __slots__ = ("shared", "shared_bytes", "bepoch", "wepoch")

    def __init__(self, nwarps):
        self.shared: dict = {}
        self.shared_bytes = 0
        self.bepoch = 0
        self.wepoch = [0] * nwarps


class Thread:
    """kernel 的第一个参数：threadIdx、blockIdx、blockDim、gridDim，以及共享内存、屏障、warp 操作、原子操作。"""

    def __init__(self, tidx: Dim3, bidx: Dim3, bdim: Dim3, gdim: Dim3, blk: _Block):
        self.threadIdx, self.blockIdx, self.blockDim, self.gridDim = tidx, bidx, bdim, gdim
        self.tid = tidx.x + tidx.y * bdim.x + tidx.z * bdim.x * bdim.y
        self.block_linear = bidx.x + bidx.y * gdim.x + bidx.z * gdim.x * gdim.y
        self.gtid = self.block_linear * bdim.size + self.tid
        self.lane = self.tid % WARP_SIZE
        self.warp_in_block = self.tid // WARP_SIZE
        self.warp_gid = self.block_linear * cdiv(bdim.size, WARP_SIZE) + self.warp_in_block
        self.blk = blk
        self.log: list = []
        self.pending_sync = False

    @property
    def warp_id(self):
        return self.warp_in_block

    def describe(self):
        t, b = self.threadIdx, self.blockIdx
        tt = f"{t.x}" if self.blockDim.y == 1 and self.blockDim.z == 1 else f"({t.x},{t.y},{t.z})"
        bb = f"{b.x}" if self.gridDim.y == 1 and self.gridDim.z == 1 else f"({b.x},{b.y},{b.z})"
        return f"线程 threadIdx={tt} blockIdx={bb}"

    # 共享内存
    def shared(self, name: str, shape, dtype=np.float32) -> SharedArray:
        arr = self.blk.shared.get(name)
        if arr is None:
            data = np.full(shape, np.nan, dtype=dtype) if np.dtype(dtype).kind == "f" else np.full(shape, -7, dtype=dtype)
            self.blk.shared_bytes += data.nbytes
            if self.blk.shared_bytes > MAX_SHARED_BYTES:
                raise KernelError(f"每个 block 的共享内存超过了 {MAX_SHARED_BYTES // 1024} KB")
            arr = SharedArray(data, name, initialized=False)
            arr.base = 0 if not self.blk.shared else max(a.base + a.data.nbytes for a in self.blk.shared.values())
            self.blk.shared[name] = arr
        elif (tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)) != arr.shape:
            raise KernelError(f"共享数组 {name} 的形状前后不一致")
        return arr

    # 屏障与 warp 操作：都要 yield
    def syncthreads(self):
        self.pending_sync = True
        return _Sync()

    def syncwarp(self):
        return _WarpOp("syncwarp")

    def shfl_down(self, value, delta, width=WARP_SIZE):
        return _WarpOp("down", value, int(delta), width)

    def shfl_up(self, value, delta, width=WARP_SIZE):
        return _WarpOp("up", value, int(delta), width)

    def shfl_xor(self, value, mask, width=WARP_SIZE):
        return _WarpOp("xor", value, int(mask), width)

    def shfl(self, value, src_lane, width=WARP_SIZE):
        return _WarpOp("idx", value, int(src_lane), width)

    def ballot(self, pred):
        return _WarpOp("ballot", bool(pred))

    def any(self, pred):
        return _WarpOp("any", bool(pred))

    def all(self, pred):
        return _WarpOp("all", bool(pred))

    # 原子操作：不需要 yield，返回旧值
    def _atomic(self, arr: DeviceArray, idx, fn):
        if not isinstance(arr, DeviceArray):
            raise KernelError("原子操作的第一个参数必须是设备数组或共享数组")
        i = arr._index(idx)
        arr._record(i, True, atomic=True)
        old = arr.flat[i]
        arr.flat[i] = fn(old)
        return old

    def atomic_add(self, arr, idx, value):
        return self._atomic(arr, idx, lambda old: old + value)

    def atomic_max(self, arr, idx, value):
        return self._atomic(arr, idx, lambda old: max(old, value))

    def atomic_min(self, arr, idx, value):
        return self._atomic(arr, idx, lambda old: min(old, value))

    def atomic_exch(self, arr, idx, value):
        return self._atomic(arr, idx, lambda old: value)

    def atomic_cas(self, arr, idx, compare, value):
        return self._atomic(arr, idx, lambda old: value if old == compare else old)


# ---------------------------------------------------------------- launch

def _warp_result(op: _WarpOp, lane: int, values: dict, preds: dict):
    w = op.width
    seg = lane - lane % w
    if op.kind == "down":
        src = lane + op.arg
        return values[src] if src < seg + w and src in values else values[lane]
    if op.kind == "up":
        src = lane - op.arg
        return values[src] if src >= seg and src in values else values[lane]
    if op.kind == "xor":
        src = lane ^ op.arg
        return values[src] if seg <= src < seg + w and src in values else values[lane]
    if op.kind == "idx":
        src = seg + op.arg % w
        return values.get(src, values[lane])
    if op.kind == "ballot":
        return sum(1 << l for l, p in preds.items() if p)
    if op.kind == "any":
        return any(preds.values())
    if op.kind == "all":
        return all(preds.values())
    return None


def _account(threads, stats: Stats):
    """把每个 warp 在这一阶段的访存日志按"第 k 条访存指令"对齐，统计扇区数和 bank conflict。"""
    warps: dict = {}
    for th in threads:
        if th.log:
            warps.setdefault(th.warp_in_block, []).append(th.log)
    for logs in warps.values():
        for k in range(max(len(l) for l in logs)):
            groups: dict = {}
            for l in logs:
                if k < len(l):
                    arr, write, atomic, addr = l[k]
                    groups.setdefault((id(arr), write, atomic), (arr, []))[1].append(addr)
            for (_, write, atomic), (arr, addrs) in groups.items():
                if atomic:
                    stats.atomics += len(addrs)
                    if arr.space == "global":
                        stats.global_atomics += len(addrs)
                    else:
                        stats.shared_atomics += len(addrs)
                    continue
                if arr.space == "global":
                    sectors = len({a // SECTOR for a in addrs} | {(a + arr.itemsize - 1) // SECTOR for a in addrs})
                    useful = len(set(addrs)) * arr.itemsize
                    per = stats.arrays.setdefault(arr.name, {"load_sectors": 0, "load_bytes": 0, "store_sectors": 0,
                                                             "store_bytes": 0})
                    kind = "store" if write else "load"
                    per[kind + "_sectors"] += sectors
                    per[kind + "_bytes"] += useful
                    if write:
                        stats.global_store_requests += 1
                        stats.global_store_sectors += sectors
                        stats.global_store_bytes += useful
                    else:
                        stats.global_load_requests += 1
                        stats.global_load_sectors += sectors
                        stats.global_load_bytes += useful
                else:
                    banks: dict = {}
                    for a in set(addrs):
                        word = a // 4
                        banks.setdefault(word % BANKS, set()).add(word)
                    wave = max(len(v) for v in banks.values())
                    stats.shared_wavefronts += wave
                    stats.bank_conflicts += wave - 1
                    if write:
                        stats.shared_store_requests += 1
                    else:
                        stats.shared_load_requests += 1
    for th in threads:
        th.log = []


def _run_block(fn, is_gen, bidx: Dim3, bdim: Dim3, gdim: Dim3, args, stats: Stats):
    global _CUR
    nthreads = bdim.size
    blk = _Block(cdiv(nthreads, WARP_SIZE))
    threads = []
    for z in range(bdim.z):
        for y in range(bdim.y):
            for x in range(bdim.x):
                threads.append(Thread(Dim3(x, y, z), bidx, bdim, gdim, blk))
    stats.threads += nthreads
    stats.blocks += 1
    try:
        if not is_gen:
            for th in threads:
                _CUR = th
                fn(th, *args)
                if th.pending_sync:
                    raise KernelError("用到了 t.syncthreads()，要写成 yield t.syncthreads()（kernel 因此变成生成器）")
            _account(threads, stats)
            return
        # 每个线程是一个协程，状态：ready（可以继续执行）、barrier（在等 __syncthreads）、
        # warp（在等 warp 操作）、done。不同 warp 可以停在不同的地方，和 GPU 一样：
        # warp 操作在整个 warp 到齐时完成，屏障在整个 block 到齐时放行。
        gens = {th.tid: fn(th, *args) for th in threads}
        send = {th.tid: None for th in threads}
        state = {th.tid: "ready" for th in threads}
        ops = {}
        ready = list(threads)
        by_warp: dict = {}
        for th in threads:
            by_warp.setdefault(th.warp_in_block, []).append(th)
        while True:
            for th in ready:
                _CUR = th
                try:
                    op = gens[th.tid].send(send[th.tid])
                except StopIteration:
                    if th.pending_sync:
                        raise KernelError("t.syncthreads() 要写成 yield t.syncthreads()，否则屏障不会生效") from None
                    state[th.tid] = "done"
                    continue
                if not isinstance(op, (_Sync, _WarpOp)):
                    raise KernelError(f"kernel 里只能 yield t.syncthreads() 或 warp 操作，却 yield 了 {op!r}")
                if th.pending_sync and not isinstance(op, _Sync):
                    raise KernelError("t.syncthreads() 要写成 yield t.syncthreads()，否则屏障不会生效")
                th.pending_sync = False
                ops[th.tid] = op
                state[th.tid] = "barrier" if isinstance(op, _Sync) else "warp"
            _CUR = None
            _account(ready, stats)
            ready = []
            for w, members in by_warp.items():
                waiting = [t for t in members if state[t.tid] == "warp"]
                if not waiting:
                    continue
                first = ops[waiting[0].tid]
                bad = next((t for t in members if state[t.tid] != "warp" or ops[t.tid].kind != first.kind), None)
                if bad is not None:
                    why = {"done": "已经返回", "barrier": "停在了 __syncthreads() 上"}.get(state[bad.tid], "执行的是别的 warp 操作")
                    raise KernelError(f"warp 操作必须由 warp 里所有线程一起执行：{waiting[0].describe()} 执行了 "
                                      f"{first.kind}，而 {bad.describe()} {why}（分支发散或提前 return）")
                values = {t.lane: ops[t.tid].value for t in waiting}
                for t in waiting:
                    send[t.tid] = _warp_result(ops[t.tid], t.lane, values, values)
                    state[t.tid] = "ready"
                blk.wepoch[w] += 1
                stats.warp_ops += 1
                ready.extend(waiting)
            if ready:
                continue
            alive = [t for t in threads if state[t.tid] != "done"]
            if not alive:
                break
            # 走到这里，活着的线程都停在屏障上
            if len(alive) != len(threads):
                waiting = alive[0]
                gone = next(t for t in threads if state[t.tid] == "done")
                raise KernelError(f"__syncthreads() 必须由 block 里所有线程执行：{waiting.describe()} 在等屏障，"
                                  f"而 {gone.describe()} 已经返回（例如在屏障前提前 return）")
            blk.bepoch += 1
            stats.syncthreads += 1
            for t in alive:
                send[t.tid] = None
                state[t.tid] = "ready"
            ready = alive
    finally:
        _CUR = None


def launch(fn, grid, block, *args, check_races: bool = True) -> Stats:
    """fn<<<grid, block>>>(*args)。返回这次 launch 的访存统计。"""
    fn = getattr(fn, "fn", fn)
    gdim, bdim = _dim3(grid), _dim3(block)
    if bdim.size > MAX_THREADS_PER_BLOCK:
        raise KernelError(f"每个 block 最多 {MAX_THREADS_PER_BLOCK} 个线程，这里是 {bdim.size}")
    if bdim.size <= 0 or gdim.size <= 0:
        raise KernelError(f"grid 和 block 的每一维都必须是正整数：grid={gdim}，block={bdim}")
    for a in args:
        if isinstance(a, np.ndarray):
            raise KernelError("kernel 的参数不能是 numpy 数组：先用 gs.to_device() 拷到设备上")
    stats = Stats()
    is_gen = inspect.isgeneratorfunction(fn)
    _TOUCHED.clear()
    try:
        for bz in range(gdim.z):
            for by in range(gdim.y):
                for bx in range(gdim.x):
                    _run_block(fn, is_gen, Dim3(bx, by, bz), bdim, gdim, args, stats)
    finally:
        for arr in _TOUCHED:
            arr._reset_tracking()
        _TOUCHED.clear()
    global last_stats
    last_stats = stats
    return stats


last_stats = None


class kernel:
    """装饰器：让 kernel 支持 CUDA 风格的 add[grid, block](...) 调用。"""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __getitem__(self, cfg):
        grid, block = cfg
        return lambda *args: launch(self.fn, grid, block, *args)

    def __call__(self, *args, **kwargs):
        raise KernelError(f"kernel {self.__name__} 要用 {self.__name__}[grid, block](...) 启动")
