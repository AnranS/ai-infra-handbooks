import heapq
from collections import deque


def run(tasks):
    now, seq = 0.0, 0
    ready = deque((name, gen, True) for name, gen in tasks.items())   # (名字, 生成器, 是否还没启动)
    timers, done = [], {}

    def wake_due():                             # 已经到期的定时器按到期时间、登记顺序放回就绪队列
        while timers and timers[0][0] <= now:
            _, _, name, gen = heapq.heappop(timers)
            ready.append((name, gen, False))

    while ready or timers:
        wake_due()
        if not ready:                           # 没有可以跑的：把时钟拨到最早的定时器
            now = timers[0][0]
            wake_due()
        name, gen, first = ready.popleft()
        try:
            kind, s = next(gen) if first else gen.send(None)
        except StopIteration:
            done[name] = round(now, 6)
            continue
        if kind == "sleep":                     # 登记定时器，让出
            heapq.heappush(timers, (now + s, seq, name, gen))
            seq += 1
        elif kind == "cpu":                     # 同步计算：独占时钟，做完排到队尾
            now += s
            ready.append((name, gen, False))
        else:
            raise ValueError(f"未知的请求：{kind}")
    return done
