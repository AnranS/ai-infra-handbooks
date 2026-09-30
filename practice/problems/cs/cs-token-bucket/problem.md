---
title: 令牌桶限流
chapter: net/load-balance.md
difficulty: 中等
tags: [限流, 令牌桶, 过载保护]
---
推理服务的限流要按 token 数算，而不是按请求数。实现一个令牌桶：

`TokenBucket(rate, capacity)`：每秒放 `rate` 个令牌，桶最多装 `capacity` 个（初始是满的）。

- `allow(now, cost=1)`：`now` 是单调递增的时间戳（秒，浮点）。先按距上次操作的时间补充令牌（不超过容量），如果余量 `>= cost` 就扣掉并返回 `True`，否则**不扣**返回 `False`；
- `wait_time(now, cost=1)`：还要等多久才够 `cost` 个令牌（已经够就返回 0.0）。不改变状态。

```python
b = TokenBucket(rate=10, capacity=10)
b.allow(0.0, 10)        # True：桶是满的
b.allow(0.0, 1)         # False：一个令牌都不剩
round(b.wait_time(0.0, 5), 2)   # 0.5：每秒 10 个，攒 5 个要 0.5 秒
b.allow(1.0, 5)         # True：一秒后补了 10 个（封顶），够了
```

<!-- 题解 -->
令牌桶不需要定时器：记下上次操作的时间和当时的余量，下次用到时按时间差补充 `(now - last) * rate` 个，再和容量取小。这种"按需补充"的写法既省事又精确。

两个容易写错的地方：（1）拒绝时**不能扣令牌**，否则大请求会把小请求饿死；（2）`now` 要用单调时钟（`time.monotonic()`），用墙上时钟会在校时时跳变——这和事件循环里定时器的道理一样（见 [epoll 与 io_uring](../os/io-models.md)）。

`capacity` 决定了允许多大的突发：容量等于速率时，最多允许攒一秒的量。推理服务通常同时限两样：输入 token 的速率（保护 prefill 的算力）和并发请求数（保护显存），任何一个超了就拒绝或排队。
