# 复杂度讲的是"增长"，但常数也要心里有数：本机上几种常见操作的实测耗时
import random
import time
from collections import deque

random.seed(0)
N = 100_000
data = list(range(N))
random.shuffle(data)
s, d = set(data), {x: x for x in data}
dq = deque(data)


def per_call(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


rows = [
    ("x in list（未命中）", "O(n)", per_call(lambda: -1 in data, 20)),
    ("x in set（未命中）", "O(1)", per_call(lambda: -1 in s, 200000)),
    ("d[x]", "O(1)", per_call(lambda: d[12345], 200000)),
    ("list.append", "O(1) 摊还", per_call(lambda: data.append(0) or data.pop(), 200000)),
    ("deque.popleft", "O(1)", per_call(lambda: dq.appendleft(dq.popleft()), 200000)),
    ("sorted(data)", "O(n log n)", per_call(lambda: sorted(data), 20)),
]
print(f"n = {N:,} 时单次操作的耗时")
for name, big_o, sec in rows:
    print(f"  {name:22s} {big_o:12s} {sec * 1e6:9.2f} 微秒")
print()
M = 30_000
print(f"把 {M:,} 个元素当队列全部取出来：")
q = data[:M]
t0 = time.perf_counter()
while q:
    q.pop(0)                                          # 每次都要把后面的元素整体前移
t_list = time.perf_counter() - t0
q = deque(data[:M])
t0 = time.perf_counter()
while q:
    q.popleft()
t_deque = time.perf_counter() - t0
print(f"  list.pop(0)：  {t_list * 1e3:8.1f} ms（O(n²)）")
print(f"  deque.popleft：{t_deque * 1e3:8.1f} ms（O(n)），快 {t_list / t_deque:.0f} 倍")
print()
print("用错数据结构差多少：求两个 20000 元素列表的交集")
a, b = list(range(20000)), list(range(10000, 30000))
t0 = time.perf_counter()
naive = [x for x in a if x in b]                      # O(n × m)：每个元素都扫一遍另一个列表
t_naive = time.perf_counter() - t0
t0 = time.perf_counter()
fast = sorted(set(a) & set(b))                        # O(n + m)
t_fast = time.perf_counter() - t0
print(f"  列表里查找：{t_naive * 1e3:8.1f} ms")
print(f"  转成集合：  {t_fast * 1e3:8.1f} ms，快 {t_naive / t_fast:.0f} 倍；结果一致：{sorted(naive) == fast}")
