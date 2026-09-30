import bisect
import hashlib


def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


class Ring:
    def __init__(self, nodes, vnodes=100):
        self.vnodes = vnodes
        self.points = sorted((h(f"{n}#{v}"), n) for n in nodes for v in range(vnodes))
        self.positions = [p for p, _ in self.points]

    def _start(self, key):
        return bisect.bisect(self.positions, h(key))

    def route(self, key):
        i = self._start(key)
        return self.points[i % len(self.points)][1]

    def route_bounded(self, key, load, cap):
        i = self._start(key)
        seen = set()
        for step in range(len(self.points)):
            node = self.points[(i + step) % len(self.points)][1]
            if node in seen:
                continue
            if load.get(node, 0) < cap:
                return node
            seen.add(node)
        return self.route(key)

    def rebalance_ratio(self, other, keys):
        if not keys:
            return 0.0
        moved = sum(1 for k in keys if self.route(k) != other.route(k))
        return moved / len(keys)
