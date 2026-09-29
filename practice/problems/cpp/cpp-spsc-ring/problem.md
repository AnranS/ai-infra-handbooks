---
title: 单生产者单消费者无锁环形队列
chapter: concurrency/lockfree-pool.md
difficulty: 困难
tags: [无锁, 内存序, 环形队列]
sanitize: thread
---
实现 `SpscQueue<T, Cap>`：只有**一个**生产者线程调用 `try_push`，只有**一个**消费者线程调用 `try_pop`，不使用锁。

- `Cap` 是 2 的幂，队列最多同时容纳 `Cap` 个元素；
- `bool try_push(const T& v)`：满了返回 `false`，否则放入并返回 `true`；
- `bool try_pop(T& out)`：空了返回 `false`，否则取出最早放入的元素并返回 `true`；
- 元素先进先出，下标会绕回数组开头（测试会放入、取出远多于 `Cap` 个元素）。

测试在 **ThreadSanitizer** 下运行：生产者和消费者同时工作，要求不丢、不重、不乱序，并且没有数据竞争——想想读写指针分别由谁写、用什么内存序。

<!-- 题解 -->
`head`（读到哪了）只由消费者写，`tail`（写到哪了）只由生产者写，两者只增不减，用 `index & (Cap - 1)` 映射到数组下标。
生产者写完元素后用 **release** 存 `tail`，消费者用 **acquire** 读 `tail` 再读元素；反方向同理，消费者读完元素后 release 存 `head`，生产者 acquire 读 `head` 才能复用那个槽位。
两个指针放在不同的缓存行（`alignas(64)`），性能会好很多。见 [无锁队列与线程池](cpp://concurrency/lockfree-pool/)。
