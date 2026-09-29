---
title: 带 future 的线程池
chapter: concurrency/lockfree-pool.md
difficulty: 困难
tags: [线程池, future, 异常传递]
sanitize: thread
---
实现一个固定线程数的线程池 `ThreadPool`：

- `explicit ThreadPool(unsigned n)`：启动 `n` 个工作线程；
- `template <class F> auto submit(F f) -> std::future<std::invoke_result_t<F>>`：提交一个无参的任务，返回拿结果用的 `future`；任务抛出的异常要在调用者 `get()` 时重新抛出；
- 析构函数：**把已经提交的任务全部执行完**，再停止并回收所有工作线程；
- 任务要真正地并行执行（测试会同时提交 `n` 个互相等待的任务，只有它们同时在运行时才能完成）。

测试在 **ThreadSanitizer** 下运行。

<!-- 题解 -->
任务队列用"互斥锁 + 条件变量"，`submit` 用 `std::packaged_task` 包装任务（它把返回值和异常都存进 `future` 的共享状态）；
因为 `std::function` 要求可拷贝，而 `packaged_task` 只能移动，所以把它放进 `shared_ptr` 再包一层 lambda。
工作线程**在锁外执行任务**；析构时设置停止标志并 `notify_all`，工作线程在"已停止且队列为空"时才退出。
工作线程用 `std::jthread` 并**最后声明**，它们最先析构（join），此时锁和队列都还活着。见 [无锁队列与线程池](cpp://concurrency/lockfree-pool/#线程池)。
