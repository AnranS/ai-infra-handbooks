class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate, self.capacity = rate, capacity
        self.tokens = float(capacity)
        self.last = None

    def _refill(self, now):
        if self.last is None:
            self.last = now
            return
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def allow(self, now, cost=1):
        self._refill(now)
        if self.tokens >= cost:
            self.tokens -= cost                        # 只有通过时才扣
            return True
        return False

    def wait_time(self, now, cost=1):
        tokens = self.tokens
        if self.last is not None:
            tokens = min(self.capacity, tokens + (now - self.last) * self.rate)
        return 0.0 if tokens >= cost else (cost - tokens) / self.rate
