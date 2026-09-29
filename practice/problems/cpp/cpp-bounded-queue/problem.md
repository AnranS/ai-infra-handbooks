---
title: 有界阻塞队列（反压）
chapter: concurrency/threads.md
difficulty: 中等
tags: [条件变量, 生产者消费者, 反压]
sanitize: thread
---
实现一个多生产者、多消费者的有界阻塞队列 `BoundedQueue<T>`：

- `explicit BoundedQueue(std::size_t capacity)`；
- `bool push(T v)`：队列满时**阻塞**，直到有空位；队列已关闭时返回 `false`（包括阻塞期间被关闭）；
- `std::optional<T> pop()`：队列空时阻塞；队列关闭且已经取空时返回 `std::nullopt`（关闭前放进去的元素仍然要能取出来）；
- `void close()`：关闭队列，唤醒所有等待的线程；
- `std::size_t size()`：当前元素个数。

元素先进先出。测试在 **ThreadSanitizer** 下运行：多个生产者、多个消费者同时读写，所有元素恰好被取出一次；满的时候生产者必须真的阻塞。

<!-- 题解 -->
一把互斥锁、两个条件变量（"不空"给消费者等，"不满"给生产者等）。`wait` 一定带谓词：`push` 等 `size < cap || closed`，`pop` 等 `!empty || closed`；
`close` 设置标志后 `notify_all` 两个条件变量。见 [线程、锁与条件变量](cpp://concurrency/threads/) 的练习 1。
