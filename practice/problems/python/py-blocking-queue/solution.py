import threading
import time
from collections import deque


class BlockingQueue:
    def __init__(self, maxsize: int):
        if maxsize < 1:
            raise ValueError("maxsize 必须 >= 1")
        self.maxsize = maxsize
        self.items = deque()
        self.closed = False
        lock = threading.Lock()
        self.not_full = threading.Condition(lock)
        self.not_empty = threading.Condition(lock)

    @staticmethod
    def _wait(cond, pred, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not pred():
            if deadline is None:
                cond.wait()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("等待超时")
                cond.wait(remaining)

    def put(self, item, timeout=None):
        with self.not_full:
            if self.closed:
                raise RuntimeError("队列已关闭")
            self._wait(self.not_full, lambda: self.closed or len(self.items) < self.maxsize, timeout)
            if self.closed:
                raise RuntimeError("队列已关闭")
            self.items.append(item)
            self.not_empty.notify()

    def get(self, timeout=None):
        with self.not_empty:
            self._wait(self.not_empty, lambda: self.items or self.closed, timeout)
            if not self.items:
                raise EOFError("队列已关闭且为空")
            item = self.items.popleft()
            self.not_full.notify()
            return item

    def close(self):
        with self.not_full:
            self.closed = True
            self.not_full.notify_all()
            self.not_empty.notify_all()

    def __len__(self):
        with self.not_full:
            return len(self.items)
