class Ring:
    def __init__(self, n_readers, n_chunks):
        self.n_readers, self.n_chunks = n_readers, n_chunks
        self.meta = bytearray(n_chunks * (1 + n_readers))   # 每块：写标志 + 每个读者一个读标志
        self.data = [None] * n_chunks
        self.w = 0                                  # 写者下一次写哪一块（不取模的序号）
        self.r = [0] * n_readers                    # 每个读者下一次读哪一块
        self.log = []

    def _set(self, index, value):
        self.meta[index] = value
        self.log.append((index, value))

    def try_write(self, payload):
        i = self.w % self.n_chunks
        base = i * (1 + self.n_readers)
        readers = self.meta[base + 1:base + 1 + self.n_readers]
        if self.meta[base] == 1 and not all(readers):   # 写过了、还有人没读：不能覆盖
            return False
        self._set(base, 0)
        self.data[i] = payload
        for k in range(self.n_readers):             # 先清读标志……
            self._set(base + 1 + k, 0)
        self._set(base, 1)                          # ……最后才置写标志
        self.w += 1
        return True

    def try_read(self, reader):
        i = self.r[reader] % self.n_chunks
        base = i * (1 + self.n_readers)
        if self.meta[base] != 1 or self.meta[base + 1 + reader] != 0:
            return None
        payload = self.data[i]
        self._set(base + 1 + reader, 1)             # 读完才标记，写者这时才可能覆盖
        self.r[reader] += 1
        return payload
