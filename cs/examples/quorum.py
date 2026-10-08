# 法定人数（quorum）：写 W 个副本、读 R 个副本，只要 W + R > N 就一定能读到最新的写
from itertools import combinations


def always_fresh(n, w, r):
    """任取一个写集合和一个读集合，是否总有交集"""
    return all(set(a) & set(b) for a in combinations(range(n), w) for b in combinations(range(n), r))


print("N   W   R   读一定最新   能容忍几台故障（写）")
for n, w, r in [(3, 2, 2), (3, 3, 1), (3, 1, 3), (3, 1, 1), (5, 3, 3), (5, 4, 2), (5, 2, 2)]:
    print(f"{n}   {w}   {r}   {'是' if always_fresh(n, w, r) else '否':6s}      {n - w}")
print()
print("多数派（W = R = ⌊N/2⌋+1）在不同副本数下的容错能力：")
for n in (1, 3, 5, 7):
    majority = n // 2 + 1
    print(f"  {n} 个副本：多数派 {majority}，能容忍 {n - majority} 台故障")
print()
print("为什么用奇数个副本：加一台不一定提高容错")
for n in range(2, 8):
    print(f"  {n} 个副本：多数派 {n // 2 + 1}，容忍 {n - (n // 2 + 1)} 台")
