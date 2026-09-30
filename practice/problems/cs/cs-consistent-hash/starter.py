import bisect
import hashlib


def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


class Ring:
    def __init__(self, nodes, vnodes=100):
        self.vnodes = vnodes
        self.points = sorted((h(f"{n}#{v}"), n) for n in nodes for v in range(vnodes))
        self.positions = [p for p, _ in self.points]

    def route(self, key):
        i = bisect.bisect(self.positions, h(key))
        return self.points[i][1]                       # 走到环尾时越界：没有回绕

    def route_bounded(self, key, load, cap):
        node = self.route(key)
        return node if load.get(node, 0) < cap else self.points[0][1]   # 超了就都丢给第一个节点

    def rebalance_ratio(self, other, keys):
        moved = sum(1 for k in keys if self.route(k) != other.route(k))
        return moved / len(keys)
