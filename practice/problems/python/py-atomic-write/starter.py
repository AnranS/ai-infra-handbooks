import contextlib
import os
import tempfile
import time


@contextlib.contextmanager
def atomic_write(path, mode="w"):
    with open(path, mode) as f:      # 不是原子的：出错时会留下写了一半的文件
        yield f


class Timer:
    pass
