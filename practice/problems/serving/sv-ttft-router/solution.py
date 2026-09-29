class Router:
    def __init__(self, n, speed, slo, block=64):
        self.n, self.speed, self.slo, self.block = n, speed, slo, block
        self.busy = [0.0] * n
        self.cache = [set() for _ in range(n)]

    def _hit(self, i, keys):
        n = 0
        for k in keys:
            if k not in self.cache[i]:
                break
            n += 1
        return n * self.block

    def route(self, t, prefix_keys, length):
        best = None
        for i in range(self.n):
            hit = self._hit(i, prefix_keys)
            est = max(self.busy[i] - t, 0.0) + (length - hit) / self.speed
            if best is None or est < best[0]:
                best = (est, i, hit)
        est, i, hit = best
        if est > self.slo:
            return None
        self.busy[i] = max(self.busy[i], t) + (length - hit) / self.speed
        self.cache[i].update(prefix_keys)
        return i, hit, self.busy[i] - t
