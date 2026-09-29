import threading
import time
from collections import deque


class BlockingQueue:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.items = deque()

    def put(self, item, timeout=None):
        self.items.append(item)

    def get(self, timeout=None):
        return self.items.popleft()

    def close(self):
        pass

    def __len__(self):
        return len(self.items)
