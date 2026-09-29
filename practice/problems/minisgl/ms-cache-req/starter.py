from dataclasses import dataclass


@dataclass(frozen=True)
class Handle:
    cached_len: int
    key: tuple = ()


class ToyPrefixCache:
    """按页对齐的前缀缓存（简化版）：只记录"哪些前缀在缓存里、KV 在哪里"，以及每个句柄的锁计数。"""

    def __init__(self, page_size=1):
        self.ps = page_size
        self.store = {}           # 前缀（元组） -> 这一整段的 KV 位置（元组）
        self.locks = {}           # Handle -> 锁计数

    def insert_prefix(self, ids, indices):
        n = len(ids) // self.ps * self.ps
        ids, indices = tuple(ids[:n]), tuple(indices[:n])
        prefix_len = max((k for k in range(0, n + 1, self.ps) if ids[:k] in self.store or k == 0), default=0)
        known = self.store.get(ids[:prefix_len], ())
        full = known + indices[prefix_len:]
        for k in range(self.ps, n + 1, self.ps):
            self.store.setdefault(ids[:k], full[:k])
        return prefix_len, Handle(n, ids)

    def lock(self, handle):
        self.locks[handle] = self.locks.get(handle, 0) + 1

    def unlock(self, handle):
        self.locks[handle] -= 1
        assert self.locks[handle] >= 0, "解锁次数多于加锁次数"


@dataclass
class ReqState:
    input_ids: list
    cached_len: int
    page_indices: list
    handle: Handle


def cache_req(cache, req, finished, free):
    pass
