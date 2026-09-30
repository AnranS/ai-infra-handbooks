class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate, self.capacity = rate, capacity
        self.tokens = float(capacity)
        self.last = None

    def _refill(self, now):
        if self.last is None:
            self.last = now
            return
        self.tokens += (now - self.last) * self.rate   # 忘了封顶：长时间空闲后能攒出无限突发
        self.last = now

    def allow(self, now, cost=1):
        self._refill(now)
        self.tokens -= cost                            # 不管过不过都扣：大请求会把小请求饿死
        return self.tokens >= 0

    def wait_time(self, now, cost=1):
        self._refill(now)                              # 查询不该改变状态
        return 0.0 if self.tokens >= cost else (cost - self.tokens) / self.rate
