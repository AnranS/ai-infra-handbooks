import os
import select
import time

pipes = [os.pipe() for _ in range(2000)]          # 2000 个都没有数据的连接（用管道代替）
p = select.poll()
ep = select.epoll()
for r, _ in pipes:
    p.register(r, select.POLLIN)
    ep.register(r, select.EPOLLIN)
os.write(pipes[-1][1], b"x")                      # 只有最后一个"连接"来了数据


def per_call(fn, n=2000):
    t = time.perf_counter()
    for _ in range(n):
        ready = fn()
    return (time.perf_counter() - t) / n * 1e6, len(ready)


for name, fn in [("poll", lambda: p.poll(0)), ("epoll", lambda: ep.poll(0))]:
    us, ready = per_call(fn)
    print(f"{name}：2000 个连接里 {ready} 个就绪，每次调用 {us:.1f} µs")
