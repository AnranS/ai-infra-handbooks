from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = OrderedDict()                 # 最左边是最久未使用

    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)
        return self.d[key]

    def put(self, key, value):
        if self.cap <= 0:
            return
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)

    def keys_lru_first(self):
        return list(self.d)
