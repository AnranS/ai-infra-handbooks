class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = {}
        self.order = []                        # 用列表维护顺序：remove 是 O(n)

    def get(self, key):
        if key not in self.d:
            return -1
        return self.d[key]                     # 命中却没有更新使用顺序

    def put(self, key, value):
        if key not in self.d:
            self.order.append(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            oldest = self.order.pop(0)
            del self.d[oldest]

    def keys_lru_first(self):
        return list(self.order)
