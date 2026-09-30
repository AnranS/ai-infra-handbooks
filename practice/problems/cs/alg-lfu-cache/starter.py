from collections import Counter


class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.vals = {}
        self.freq = Counter()

    def get(self, key):
        if key not in self.vals:
            return -1
        self.freq[key] += 1
        return self.vals[key]

    def put(self, key, value):
        if key not in self.vals and len(self.vals) >= self.cap:
            victim = min(self.freq, key=lambda k: self.freq[k])   # 同次数时没有按时间选
            del self.vals[victim]
            del self.freq[victim]
        self.vals[key] = value
        self.freq[key] += 1                    # 容量为 0 时也会存进去

    def keys_by_freq(self):
        return sorted(((k, self.freq[k]) for k in self.vals), key=lambda t: t[1])
