# 一致性哈希环：每台机器在环上放 vnodes 个虚拟节点，键顺时针找到的第一个虚拟节点就是它的归属
import bisect
import hashlib
from collections import Counter


def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


class Ring:
    def __init__(self, nodes, vnodes=1):
        self.vnodes = vnodes
        self.points = sorted((h(f"{n}#{v}"), n) for n in nodes for v in range(vnodes))
        self.keys = [p for p, _ in self.points]

    def route(self, key):
        i = bisect.bisect(self.keys, h(key))
        return self.points[i % len(self.points)][1]


keys = [f"session-{i}" for i in range(100000)]
print("每台机器的虚拟节点数   负载最重/最轻   加一台机器要迁移的键")
for vnodes in (1, 10, 100, 500):
    before = Ring([f"node{i}" for i in range(8)], vnodes)
    after = Ring([f"node{i}" for i in range(9)], vnodes)
    load = Counter(before.route(k) for k in keys)
    moved = sum(1 for k in keys if before.route(k) != after.route(k))
    hi, lo = max(load.values()), min(load.values())
    print(f"{vnodes:12d} {hi / (len(keys) / 8):16.2f} / {lo / (len(keys) / 8):.2f} {moved / len(keys):14.1%}")
print("\n加第 9 台机器时，理想的迁移比例是 1/9 =", f"{1 / 9:.1%}")
