import heapq
import itertools


class Node:
    _ids = itertools.count()

    def __init__(self, key, value, timestamp):
        self.key, self.value = list(key), list(value)
        self.children = {}
        self.parent = None
        self.ref_count = 0
        self.timestamp = timestamp
        self.uid = next(Node._ids)


class Handle:
    def __init__(self, cached_len, node):
        self.cached_len, self.node = cached_len, node


class RadixCache:
    def __init__(self):
        self.clock = 0
        self.root = Node([], [], 0)
        self.root.ref_count = 1
        self.evictable_size = 0
        self.protected_size = 0

    def match_prefix(self, ids):
        pass

    def insert_prefix(self, ids, indices):
        pass

    def lock(self, handle):
        pass

    def unlock(self, handle):
        pass

    def evict(self, size):
        pass

    def matched_indices(self, handle):
        pass
