import math
from collections.abc import Sequence


class ArithSeq(Sequence):
    def __init__(self, start, stop, step=1):
        if step == 0:
            raise ValueError("step 不能为 0")
        self.start, self.stop, self.step = start, stop, step
        self._len = max(0, math.ceil((stop - start) / step))

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        if isinstance(key, slice):
            i, j, k = key.indices(self._len)
            n = len(range(i, j, k))
            start = self.start + i * self.step
            step = self.step * k
            return ArithSeq(start, start + n * step, step)
        if not isinstance(key, int) or isinstance(key, bool):
            raise TypeError(f"下标必须是整数或切片，收到 {type(key).__name__}")
        if key < 0:
            key += self._len
        if not 0 <= key < self._len:
            raise IndexError("ArithSeq 下标越界")
        return self.start + key * self.step

    def __repr__(self):
        return f"ArithSeq({self.start!r}, {self.stop!r}, {self.step!r})"

    def __eq__(self, other):
        if not isinstance(other, ArithSeq):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    __hash__ = None
