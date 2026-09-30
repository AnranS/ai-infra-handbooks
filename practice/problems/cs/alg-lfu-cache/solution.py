from collections import OrderedDict, defaultdict


class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.vals = {}                         # key -> (值, 次数)
        self.buckets = defaultdict(OrderedDict)   # 次数 -> 有序的键集合
        self.min_freq = 0

    def _touch(self, key):
        value, freq = self.vals[key]
        del self.buckets[freq][key]
        if not self.buckets[freq]:
            del self.buckets[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        self.buckets[freq + 1][key] = True
        self.vals[key] = (value, freq + 1)
        return value

    def get(self, key):
        if key not in self.vals:
            return -1
        return self._touch(key)

    def put(self, key, value):
        if self.cap <= 0:
            return
        if key in self.vals:
            self._touch(key)
            self.vals[key] = (value, self.vals[key][1])
            return
        if len(self.vals) >= self.cap:
            old, _ = self.buckets[self.min_freq].popitem(last=False)   # 最久未使用
            if not self.buckets[self.min_freq]:
                del self.buckets[self.min_freq]
            del self.vals[old]
        self.vals[key] = (value, 1)
        self.buckets[1][key] = True
        self.min_freq = 1

    def keys_by_freq(self):
        out = []
        for freq in sorted(self.buckets):
            out.extend((k, freq) for k in self.buckets[freq])
        return out
