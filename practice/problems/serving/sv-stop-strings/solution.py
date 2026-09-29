class StopChecker:
    def __init__(self, stops, include_stop=False):
        self.stops = [s for s in stops if s]
        self.include_stop = include_stop
        self.max_len = max((len(s) for s in self.stops), default=0)
        self.buf = ""
        self.done = False

    def feed(self, delta):
        if self.done:
            return "", True
        self.buf += delta
        best = None
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0 and (best is None or i < best[0] or (i == best[0] and len(s) > len(best[1]))):
                best = (i, s)
        if best is not None:
            i, s = best
            out = self.buf[: i + len(s)] if self.include_stop else self.buf[:i]
            self.buf, self.done = "", True
            return out, True
        keep = 0
        for k in range(min(len(self.buf), self.max_len - 1), 0, -1):
            tail = self.buf[-k:]
            if any(s.startswith(tail) for s in self.stops):
                keep = k
                break
        out = self.buf[: len(self.buf) - keep]
        self.buf = self.buf[len(self.buf) - keep:]
        return out, False

    def flush(self):
        if self.done:
            return ""
        out, self.buf = self.buf, ""
        return out
