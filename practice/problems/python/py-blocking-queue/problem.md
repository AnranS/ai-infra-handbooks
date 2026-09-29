---
title: 用 Condition 实现有界阻塞队列
chapter: concurrency/threads-processes.md
difficulty: 中等
tags: [threading, Condition, 生产者消费者]
requires: [local]
---
推理服务里，接收请求的线程和调度线程之间常用一个**有界阻塞队列**：队列满时生产者等待（背压），队列空时消费者等待。
不用 `queue.Queue`，用 `threading.Condition` 自己实现 `BlockingQueue(maxsize)`：

- `put(item, timeout=None)`：队列满时阻塞，直到有空位；超过 `timeout` 秒仍然没空位就抛出 `TimeoutError`；
- `get(timeout=None)`：队列空时阻塞，直到有元素；超时抛出 `TimeoutError`；先进先出；
- `close()`：关闭队列。关闭后 `put` 立即抛出 `RuntimeError`；`get` 把剩下的元素取完之后，再调用会抛出 `EOFError`（而不是永远阻塞）；正在等待的线程也要被唤醒；
- `len(q)`：当前元素个数。

> 这道题用到线程，浏览器里的 Python 不支持，请在本地判题（macOS、WSL2 都可以）。

<!-- 题解 -->
一把锁、两个条件变量：`not_full` 和 `not_empty`，共享同一把 `threading.Lock`。

- `put`：`with self.not_full:`，`while` 队满且未关闭时 `wait(剩余时间)`；醒来后必须重新检查条件（虚假唤醒、被别的生产者抢先）；放入元素后 `self.not_empty.notify()`；
- `get` 对称；
- 超时要按"截止时间"算：`deadline = time.monotonic() + timeout`，每次 `wait(deadline - now)`，否则多次被唤醒会让总等待时间超过 `timeout`；
- `close` 里 `notify_all` 两个条件变量，唤醒所有等待者。
