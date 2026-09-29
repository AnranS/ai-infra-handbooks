import contextlib
import os
import tempfile
import time


@contextlib.contextmanager
def atomic_write(path, mode="w"):
    if mode not in ("w", "wb"):
        raise ValueError(f"mode 只能是 'w' 或 'wb'，收到 {mode!r}")
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, mode) as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise


class Timer:
    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end = time.perf_counter()
        return None

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start
